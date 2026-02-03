import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from dataloader import FineWebEduDataLoader
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import math
from flash_attn import flash_attn_func
from model import Model

steps = 10000
print_every = 50
grad_accum_steps = 16
log_dir = "runs/simple/6"
lr = 6e-4


def get_lr(step):
    # Linear warmup for first 100 steps, then cosine decay
    warmup_steps = 100
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 + 0.1

torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = FineWebEduDataLoader(tokenizer, subset="sample-10BT", num_val_documents=100)
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1)



# --- Training Setup ---

# Reduced dim to 512 for faster feedback, but 1024 works too
model = Model(
    len(tokenizer), 
    512, 
    4, 
    4
) 
model.to("cuda")
adam_parameters = []
muon_parameters = []
for n, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in n:
        muon_parameters.append(p)
    else:
        adam_parameters.append(p)
# Use standard Weight Decay
optimizer1 = optim.AdamW(adam_parameters, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
optimizer2 = optim.Muon(muon_parameters, lr = lr)
optimizers = [optimizer1, optimizer2]

writer = SummaryWriter(log_dir)
start_time = time.time()
losses = []
train_iter = iter(train_dataloader)

model.compile()
epoch = 0
start_time = time.time()
for step in range(steps):
    lr_now = lr * get_lr(step)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        
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
    
    for optimizer in optimizers:
        optimizer.step()
    model.zero_grad()

    losses.append(loss_accum)
    writer.add_scalar("loss", loss_accum, step)
    
    if step % print_every == 0:
        print(f"step: {step} | epoch: {epoch} | loss: {np.mean(losses[-print_every:]):.4f} | lr: {lr_now:.2e} | avg time: {(time.time() - start_time)/(step+1):.3f}s")


state_dict = {
    "model" : model.state_dict(),
    "optimizers" : [optimizer.state_dict() for optimizer in optimizers]
}
torch.save(state_dict, "checkpoint.pt")