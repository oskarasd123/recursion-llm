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
    num_layers=8,
    num_heads=4,
    window_size=128,
)

state_dict = torch.load("runs/simple/22/checkpoint.pt")
model.load_state_dict(state_dict["model"])

model.to("cuda")

prev_prompt = "The sin function"
while True:
    text = input("prompt: ")
    if text == "r":
        text = prev_prompt
    prev_prompt = text
    ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()

    for i in range(80):
        logits = model(ids)
        new_id = torch.distributions.Categorical(logits=logits[:,-1]).sample().unsqueeze(0)
        ids = torch.cat([ids, new_id], 1)
    
    text = tokenizer.decode(ids.squeeze(0))
    print(text)

