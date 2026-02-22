import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from dataloader import FineWebDataLoader
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import math
from model import Model
import json

steps = 5000
val_every = 200
grad_accum_steps = 128
log_dir = "runs/simple/23"
lr = 1e-2
load_checkpoint = False

def get_lr(step):
    # Linear warmup, then cosine decay
    warmup_steps = 100
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (cos**2) * 0.8 + 0.2

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = FineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, num_val_documents=500)
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
test_dataloader = DataLoader(dataset.val_data, batch_size=1, num_workers=1)

model = Model(
    num_embeddings=len(tokenizer),
    dim=128*4,
    num_layers=8,
    num_heads=4,
    window_size=128,
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
    if p.ndim == 2 and "embed" not in n:
        muon_parameters.append(p)
    else:
        if "embed" not in n:
            adam_parameters.append(p)
        else:
            embed_params.append(p)
# Use standard Weight Decay
optimizer1 = optim.AdamW(adam_parameters, lr = lr, betas=(0.9, 0.95), weight_decay=0.1)
optimizer2 = optim.AdamW(embed_params, lr = lr, betas=(0.9, 0.95), weight_decay=0.1)
optimizer3 = optim.Muon(muon_parameters, lr = lr, momentum=0.8)
optimizers = [optimizer1, optimizer2, optimizer3]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

if load_checkpoint:
    state_dict = torch.load("checkpoint.pt")
    model.load_state_dict(state_dict["model"])
    for i in range(len(optimizers)):
        optimizers[i].load_state_dict(state_dict["optimizers"][i])
    print("loaded checkpoint")


writer = SummaryWriter(log_dir)
start_time = time.time()
losses = []
train_iter = iter(train_dataloader)

model.compile()
epoch = 0
start_time = time.time()
for step in range(steps):
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
        
    loss_accum = 0
    for i in range(grad_accum_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            epoch += 1
            
        ids = batch["input_ids"].cuda()
        logits = model(ids[:,:-1])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:,1:].reshape(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        loss_accum += loss.item()
    
    for opt in optimizers:
        opt.step()
    model.zero_grad()

    losses.append(loss_accum)
    writer.add_scalar("loss", loss_accum, step*grad_accum_steps)
    
    if step % val_every == 0:
        val_loss = 0
        val_examples = 0
        for batch in test_dataloader:
            ids = batch["input_ids"].cuda()
            logits = model(ids[:,:-1])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:,1:].reshape(-1))
            val_loss += loss.item()
            val_examples += 1
        val_loss /= val_examples
        writer.add_scalar("val/loss", val_loss, step*grad_accum_steps)
        print(f"step: {step} | epoch: {epoch} | loss: {np.mean(losses[-val_every:]):.4f} | val loss {val_loss:.4f} | lr mult: {get_lr(step):.2f} | avg time: {(time.time() - start_time)/(step+1):.3f}s")
    if step == 100:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

# final eval
val_loss = 0
val_examples = 0
for batch in test_dataloader:
    ids = batch["input_ids"].cuda()
    logits = model(ids[:,:-1])
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:,1:].reshape(-1))
    val_loss += loss.item()
    val_examples += 1
val_loss /= val_examples
writer.add_scalar("val/loss", val_loss, step*grad_accum_steps)
print(f"step: {step} | epoch: {epoch} | loss: {np.mean(losses[-val_every:]):.4f} | val loss {val_loss:.4f} | lr mult: {get_lr(step):.2f} | avg time: {(time.time() - start_time)/(step+1):.3f}s")




state_dict = {
    "model" : model.state_dict(),
    "optimizers" : [optimizer.state_dict() for optimizer in optimizers]
}
torch.save(state_dict, f"{log_dir}/checkpoint.pt")

json.dump({
    "hparams": {
        "num_layers" : model.num_layers,
        "dim" : model.dim,
        "num_heads" : model.num_heads,
        "window_size" : model.window_size,
    },
    "metrics" : {
        "avg_loss" : np.mean(losses),
        "val_loss" : val_loss,
    }
}, open(f"{log_dir}/metrics.json", "w"))
