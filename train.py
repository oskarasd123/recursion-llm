import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from dataloader import FineWebDataLoader, MaxLenFineWebDataLoader
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import math
from model import Model
import json
torch._dynamo.config.capture_scalar_outputs = True
#torch.autograd.set_detect_anomaly(True)

steps = 10000
val_every = 200
grad_accum_steps = 16
batch_size = 8192
start_lr = 1e-2
lr = 0.3e-2
load_checkpoint = False

log_dir = "runs/"

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
dir_index = max(list(map(int, os.listdir(log_dir))) + [-1]) + (0 if load_checkpoint else 1) # auto increment
log_dir = f"{log_dir}{dir_index}/"

def get_lr(step):
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

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=batch_size, num_val_documents=10000)
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
test_dataloader = DataLoader(dataset.val_data, batch_size=1, num_workers=1)

model = Model(
    num_embeddings=len(tokenizer),
    dim=128*4,
    num_layers=12,
    num_heads=4,
    window_size=512,
    pairs=1,
    max_seq_len=batch_size,
)
model.to("cuda")

model_numel = 0
embed_numel = 0
model_size = 0
for n, p in model.named_parameters():
    if "embed" in n:
        embed_numel += p.numel()
    else:
        model_numel += p.numel()
    model_size += p.numel() * p.element_size()
print(f"model numel: {model_numel/1000_000:.1f}M")
print(f"embed numel: {embed_numel/1000_000:.1f}M")
print(f"model size in bytes: {model_size/1024**2:.1f}MiB")



adam_parameters = []
muon_parameters = []
embed_params = []
for n, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in n and "gate" not in n:
        muon_parameters.append(p)
    else:
        if "embed" not in n:
            adam_parameters.append(p)
        else:
            embed_params.append(p)

optimizer1 = optim.AdamW(adam_parameters, lr = start_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
optimizer2 = optim.AdamW(embed_params, lr = start_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
#optimizer3 = optim.Muon(muon_parameters, lr = start_lr, momentum=0.8)
optimizer3 = optim.AdamW(muon_parameters, lr = start_lr, betas=(0.9, 0.95)) # no muon in torch==2.8.0
optimizers = [optimizer1, optimizer2, optimizer3]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]



writer = SummaryWriter(log_dir)
start_time = time.time()
losses = []
train_iter = iter(train_dataloader)


#profile_steps = 20
#
#def trace_handler(prof: torch.profiler.profile):
#    prof.export_chrome_trace(f"single-chrome-trace.json.gz")
#
#prof_ctx = torch.profiler.profile(
#    activities=[
#        # profile activity on the CPU and GPU
#        torch.profiler.ProfilerActivity.CPU,
#        torch.profiler.ProfilerActivity.CUDA,
#    ],
#    # Setup the profiler schedule to wait 5 steps, warmup for 5 steps,
#    # then activate for the remaining steps.
#    schedule=torch.profiler.schedule(wait=10, warmup=5, active=profile_steps - 15),
#    # This callback will be fired when the trace files are ready
#    on_trace_ready=trace_handler,
#    # Records the file and line number for the operation.
#    # Disabling this mainly to make the traces less cluttered
#    with_stack=True,
#    record_shapes=True,
#)

model.train()
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

start_time = time.time()
try:
    for step in range(step, steps):
        if step == 1: # first step takes more time because of compilation
            start_time = time.time() # set start time after 1st step
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        
        output_weight_means = []
        loss_accum = 0
        for i in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
                epoch += 1
                
            ids = batch["input_ids"].to("cuda", non_blocking=True)
            cu_seqlens = batch["cu_seqlens"].to("cuda", non_blocking=True).squeeze(0)
            max_seqlen = batch["max_seqlen"].item()
            logits, output_weights = model_opt(ids, cu_seqlens, max_seqlen, get_loop_steps(step), return_output_weights=True)

            token_loss = F.cross_entropy(logits[:, :-1, :].view(-1, logits.size(-1)), ids[:, 1:].reshape(-1))

            output_weight_mean = output_weights.flatten(0,-2).mean(0)
            output_weight_means.append(output_weight_mean)
            output_weight_mean = output_weight_mean * 0.999 + 0.0005
            gate_regularisation_loss = -torch.log(output_weight_mean).mean() # without this term the end gate output for the first loop would go to 1 and stop learning
            loss = token_loss + gate_regularisation_loss * 0.005
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += token_loss.item() / grad_accum_steps
            documents += len(batch["texts"])
            
        
        for opt in optimizers:
            opt.step()
        model.zero_grad()

        output_weight_means = torch.stack(output_weight_means).to("cpu", torch.float32).mean(0).numpy(force=True)
        for i, v in enumerate(output_weight_means):
            writer.add_scalar(f"output_mean/{i}", v, documents)
        losses.append(loss_accum)
        writer.add_scalar("loss", loss_accum, documents)
        
        if step % val_every == 0:
            with torch.no_grad():
                val_loss = 0
                val_examples = 0
                for batch in test_dataloader:
                    ids = batch["input_ids"].cuda()
                    cu_seqlens = batch["cu_seqlens"].cuda().squeeze(0)
                    max_seqlen = batch["max_seqlen"].item()
                    logits = model_opt(ids, cu_seqlens, max_seqlen, get_loop_steps(step))[:, :-1, :]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
                    val_loss += loss.item()
                    val_examples += 1
                val_loss /= val_examples
                writer.add_scalar("val/loss", val_loss, documents)
                print(f"step: {step} | epoch: {epoch} | loss: {np.mean(losses[-val_every:]):.4f} | val loss {val_loss:.4f} | lr mult: {get_lr(step):.2f} | avg step time: {(time.time() - start_time)/(step+1):.3f}s")
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


state_dict = {
    "model" : model.state_dict(),
    "optimizers" : [optimizer.state_dict() for optimizer in optimizers]
}
torch.save(state_dict, f"{log_dir}/checkpoint.pt")
print("model saved")

json.dump({
    "hparams": {
        "model params":{
            "num_layers" : model.num_layers,
            "dim" : model.dim,
            "num_heads" : model.num_heads,
            "window_size" : model.window_size,
            "pairs" : model.pairs,
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
