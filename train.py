import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress tensorflow warnings
import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import muon # no muon in torch==2.8.0
from dataloader import FineWebDataLoader, MaxLenFineWebDataLoader
from tokenizer_compressor import create_token_compression_map
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import math
from model import Model
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch._dynamo.config.capture_scalar_outputs = True
#torch.autograd.set_detect_anomaly(True)

# most hparams are here
steps = 10000
base_grad_accum_steps = 32
batch_size = 4096
start_lr = 0.5e-2
lr = 0.2e-2
load_checkpoint = False
val_every = 200
validate = True
log_dir = "runs/"


dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
master_process = rank == 0

assert base_grad_accum_steps%world_size == 0 # grad_accum_steps must be divisible by world_size
base_grad_accum_steps = base_grad_accum_steps//world_size


def get_lr(step): # learning rate multiplier applied on start_lr
    # Linear warmup, then cosine decay
    warmup_steps = 100
    schedule_steps = 10000
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (schedule_steps - warmup_steps)
    progress = max(min(progress, 1), 0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    end_ratio = lr/start_lr
    return (cos**3) * (1-end_ratio) + end_ratio

def get_loop_steps(step):
    frac = step/steps
    return 1

def get_grad_accum_steps(step):
    return base_grad_accum_steps
    if step < 2000:
        return base_grad_accum_steps
    if step < 3000:
        return base_grad_accum_steps*2
    return base_grad_accum_steps*4

def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
# auto increment log_dir
dir_index = max(list(map(int, os.listdir(log_dir))) + [-1]) + (0 if load_checkpoint else 1)
log_dir = f"{log_dir}{dir_index}/"
print0(f"using logdir: {log_dir}")

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=batch_size, num_val_documents=10000)
val_dataset = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=batch_size, num_val_documents=10000, val=True)
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
test_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=1)

model = Model(
    num_embeddings=len(tokenizer),
    dim=128*4,
    num_layers=12,
    num_heads=4,
    window_size=256,
    max_seq_len=batch_size,
).to(device)

#ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
model_numel = 0
embed_numel = 0
model_size = 0
for n, p in model.named_parameters():
    if "embed" in n:
        embed_numel += p.numel()
    else:
        model_numel += p.numel()
    model_size += p.numel() * p.element_size()
print0(f"model numel: {model_numel/1000_000:.1f}M")
print0(f"embed numel: {embed_numel/1000_000:.1f}M")
print0(f"model size in bytes: {model_size/1024**2:.1f}MiB")



adam_parameters = []
muon_parameters = []
embed_params = []
engram_params = []
for n, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in n and "gate" not in n:
        muon_parameters.append(p)
    else:
        if "embed" not in n:
            adam_parameters.append(p)
        else:
            if "engram" in n:
                engram_params.append(p)
            else:
                embed_params.append(p)

optimizer1 = optim.AdamW(adam_parameters, lr = start_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
optimizer2 = optim.AdamW(embed_params, lr = start_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
optimizer3 = muon.Muon(muon_parameters, lr = start_lr, momentum=0.8)
optimizer4 = optim.SparseAdam(engram_params, lr = start_lr, betas=(0.9, 0.999))
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def dense_to_sparse_gradient(p : Tensor):
    """
    Converts a dense gradient tensor into a sparse COO tensor
    based on rows that contain non-zero values.
    """
    if p.grad is None:
        return
    
    grad = p.grad
    row_mask = (grad != 0).any(dim=1)
    indices = row_mask.nonzero().flatten()

    values = grad[indices]
    sparse_grad = torch.sparse_coo_tensor(
        indices.unsqueeze(0), 
        values, 
        grad.shape
    )
    p.grad = sparse_grad


#def trace_handler(prof: torch.profiler.profile):
#    prof.export_chrome_trace(f"single-chrome-trace.json.gz")
#
#prof_ctx = torch.profiler.profile(
#    activities=[
#        # profile activity on the CPU and GPU
#        torch.profiler.ProfilerActivity.CPU,
#        torch.profiler.ProfilerActivity.CUDA,
#    ],
#    schedule=torch.profiler.schedule(wait=10, warmup=5, active=5),
#    on_trace_ready=trace_handler,
#    with_stack=True,
#    record_shapes=True,
#)

if master_process:
    writer = SummaryWriter(log_dir)
losses = []
train_iter = iter(train_dataloader)


#model.compile()
model_opt = model
model_opt = torch.compile(model, dynamic=True)
#prof_ctx.__enter__()
epoch = 0
documents = 0
step = 0

if load_checkpoint:
    state_dict = torch.load(f"{log_dir}/checkpoint.pt")
    model.load_state_dict(state_dict["model"])
    for opt, state in zip(optimizers, state_dict["optimizers"]):
        opt.load_state_dict(state)
    metrics = json.load(open(f"{log_dir}/metrics.json", "r"))
    step = metrics["hparams"]["step"]
    documents = metrics["hparams"]["documents"]

def save_model():
    if master_process:
        state_dict = {
            "model" : model.state_dict(),
            "optimizers" : [optimizer.state_dict() for optimizer in optimizers]
        }
        torch.save(state_dict, f"{log_dir}/checkpoint.pt")

        json.dump({
            "hparams": {
                "model params":{
                    "num_layers" : model.num_layers,
                    "dim" : model.dim,
                    "num_heads" : model.num_heads,
                    "window_size" : model.window_size
                },
                "steps" : steps,
                "step" : step,
                "documents" : documents,
                "lr" : lr,
                "batch_size" : batch_size,
                "grad_accum_steps" : grad_accum_steps,
                "final loop_steps" : get_loop_steps(step),
            },
            "metrics" : {
                "avg_loss" : np.mean(losses[-val_every:]),
                "val_loss" : val_loss,
            }
        }, open(f"{log_dir}/metrics.json", "w"))

start_time = time.time()
try:
    for step in range(step, steps):
        if step == 1: # first step takes more time because of compilation
            start_time = time.time() # set start time after 1st step
        grad_accum_steps = get_grad_accum_steps(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        output_weight_means = []
        loss_accum = 0
        #no_sync_ctx = ddp_model.no_sync()
        #no_sync_ctx.__enter__() # don't sync gradients except on the last backward pass
        for i in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
                epoch += 1
            #if i == grad_accum_steps-1:
            #    no_sync_ctx.__exit__(None, None, None)
                
            ids = batch["input_ids"].to(device, non_blocking=True)
            cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True).squeeze(0)
            max_seqlen = batch["max_seqlen"].item()
            logits, output_weights = model_opt(ids, cu_seqlens = cu_seqlens, max_seqlen = max_seqlen, loop_steps=get_loop_steps(step), return_output_weights=True)

            token_loss = F.cross_entropy(logits[:, :-1, :].view(-1, logits.size(-1)), ids[:, 1:].reshape(-1))

            output_weight_mean = output_weights.flatten(0,-2).mean(0)
            output_weight_mean = output_weight_mean * 0.999 + 0.0005
            gate_regularisation_loss = -torch.log(output_weight_mean).mean() # without this term the end gate output for the first loop would go to 1 and stop learning
            loss = token_loss + gate_regularisation_loss * 0.005
            loss = loss / grad_accum_steps
            loss.backward()
            output_weight_means.append(output_weight_mean.to(torch.float32).numpy(force=True))
            loss_accum += token_loss.item() / grad_accum_steps
            documents += len(batch["texts"]) * world_size
        
        handles = []
        for param in model.parameters():
            if param.grad is not None:
                handles.append(dist.all_reduce(param.grad, async_op=True))
        [handle.wait() for handle in handles]
        
        [dense_to_sparse_gradient(param) for param in engram_params]
        for opt in optimizers:
            opt.step()
        model.zero_grad()

        if master_process:
            output_weight_means = np.mean(output_weight_means, axis=0)
            for i, v in enumerate(output_weight_means):
                writer.add_scalar(f"output_mean/{i}", v, documents)
            losses.append(loss_accum)
            writer.add_scalar("loss", loss_accum, documents)
            print("flushing")
            writer.flush()
            print("step done")
        
        if step % val_every == 0:
            val_loss = 0
            val_examples = 0
            if validate:
                with torch.no_grad():
                    for batch in test_dataloader:
                        ids = batch["input_ids"].to(device, non_blocking=True)
                        cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True).squeeze(0)
                        max_seqlen = batch["max_seqlen"].item()
                        logits, output_weights = model_opt(ids, cu_seqlens = cu_seqlens, max_seqlen = max_seqlen, loop_steps=get_loop_steps(step), return_output_weights=True)
                        loss = F.cross_entropy(logits[:, :-1, :].view(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
                        dist.reduce(loss, 0)
                        val_loss += loss.item() / world_size
                        val_examples += 1
                    if master_process:
                        writer.add_scalar("val/loss", val_loss, documents)
            print0(f"step: {step} | epoch: {epoch} | loss: {np.mean(losses[-val_every:]):.4f} | val loss {val_loss:.4f} | lr mult: {get_lr(step):.2f} | avg step time: {(time.time() - start_time)/(step+1):.3f}s | train documents: {documents}")
            save_model()
        #prof_ctx.step()
    #prof_ctx.__exit__(None, None, None)
except KeyboardInterrupt:
    if step < 10:
        exit()
except torch.OutOfMemoryError as e:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(e)
save_model()
dist.destroy_process_group()