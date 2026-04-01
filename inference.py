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
import readline
import atexit
import os
import json
torch._dynamo.config.capture_scalar_outputs = True

model_path = "./runs/8/"


HISTORY_FILE = "model_prompts.history"
HISTORY_SIZE = 1000

if os.path.exists(HISTORY_FILE):
    readline.read_history_file(HISTORY_FILE)
readline.set_history_length(HISTORY_SIZE)
atexit.register(readline.write_history_file, HISTORY_FILE)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

hparams : dict = json.load(open(f"{model_path}metrics.json"))["hparams"]
model_params : dict = hparams["model params"]

model = Model(
    num_embeddings=len(tokenizer),
    dim=model_params["dim"],
    num_layers=model_params["num_layers"],
    num_heads=model_params["num_heads"],
    window_size=model_params["window_size"],
    pairs=model_params.get("pairs", 1),
    max_seq_len=8192
)
state_dict = torch.load(f"{model_path}checkpoint.pt", mmap=True)
model.load_state_dict(state_dict["model"], strict=False)


print(hparams)

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




model.to("cuda")
model.eval()
model_opt = model
#model_opt = torch.compile(model, dynamic=True)


try:
    while True:
        text = input("\033[1;32;40m$\033[0;0;0mprompt: ")
        loop_steps = hparams.get("final loop_steps", 1)
        if "%" in text:
            try:
                loop_steps = int(text.split("%")[0])
                text = text.split("%")[1]
            except:
                pass
        if not text == "c":
            prev_prompt = text
            ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda().squeeze(0)
            new_ids = []
            print(text, end="")
        else:
            print("\033[F", end='') # move cursor to the start of the previous line
        for i in range(80):
            logits = model_opt(ids.unsqueeze(0),
                        loop_steps=loop_steps,
            ).squeeze(0)
            new_id = torch.distributions.Categorical(logits=logits[-1]).sample()
            ids = torch.cat([ids, new_id.unsqueeze(0)], 0)
            new_ids.append(new_id.item())
            if new_id.item() == tokenizer.eos_token_id:
                print("<eos_token>")
                break
            try: 
                new_text = tokenizer.decode(new_ids).encode().decode() # if the string doesn't contain errors
                assert "�" not in new_text
                print(new_text, end="", flush=True)
                new_ids = []
            except Exception as e:
                if len(new_ids) > 4:
                    # print one token
                    token = new_ids.pop(0)
                    print(tokenizer.decode([token]), end="")
        print()
except KeyboardInterrupt:
    print("^C")
except EOFError:
    print("^D")

