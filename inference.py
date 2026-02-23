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



tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


model = Model(
    num_embeddings=len(tokenizer),
    dim=128*4,
    num_layers=12,
    num_heads=4,
    window_size=128,
)

state_dict = torch.load("runs/simple/23/checkpoint.pt")
model.load_state_dict(state_dict["model"])

model.to("cuda")

prev_prompt = "The sin function"
while True:
    text = input("prompt: ")
    if text == "r":
        text = prev_prompt
    prev_prompt = text
    ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda().squeeze(0)

    for i in range(80):
        logits = model(ids, torch.tensor([0, ids.shape[0]], dtype=torch.int32).cuda(), ids.shape[0])
        new_id = torch.distributions.Categorical(logits=logits[-1]).sample()
        ids = torch.cat([ids, new_id.unsqueeze(0)], 0)
    
    text = tokenizer.decode(ids)
    print(text)

