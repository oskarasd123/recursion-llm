import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from model import Model
import readline
import atexit
import os
import json
import argparse
torch._dynamo.config.capture_scalar_outputs = True

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", type=str)
args = parser.parse_args()

model_path = f"./runs/{args.run}/"


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
model_opt = torch.compile(model, dynamic=True)

def color_bg(text, r, g, b):
    return f"\x1b[48;2;{int(r)};{int(g)};{int(b)}m{text}\x1b[0m"

while True:
    try:
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
            new_output_weight_depths = []
            print(text, end="")
        else:
            print("\033[F", end='') # move cursor to the start of the previous line
        
        def stop_condition(ids : Tensor):
            return ids[0,-1].item() == tokenizer.eos_token_id
                
        inference = model_opt.generate(
            ids.unsqueeze(0),
            loop_steps=loop_steps,
            return_output_weights=True,
            stop_condition=stop_condition,
        )
        
        while True:
            try:
                new_id, output_weights = next(inference)
            except StopIteration:
                break
            #ids = torch.cat([ids, new_id.squeeze(0)], 0)
            output_weight_depth = (torch.arange(loop_steps, device="cuda")[None, None, :] * output_weights).sum(-1)[0, -1].item()
            new_ids.append(new_id.squeeze(0).item())
            new_output_weight_depths.append(output_weight_depth)
            try: 
                new_text = tokenizer.decode(new_ids).encode().decode() # if the string doesn't contain errors
                assert "�" not in new_text
                if loop_steps > 1:
                    t = np.mean(new_output_weight_depths) / (loop_steps-1)
                    r = 0 if len(new_ids) == 1 else 128
                    g = (1-t)**0.5*255
                    b = t**0.5*255
                else:
                    r = 0
                    b = 180
                    g = 180
                if new_text == "\n":
                    new_text = " "
                    print(color_bg(new_text, r, g, b), end="", flush=True)
                    print()
                else:
                    print(color_bg(new_text, r, g, b), end="", flush=True)
                new_ids = []
                new_output_weight_depths = []
            except Exception as e:
                if len(new_ids) > 4:
                    # print one token
                    token = new_ids.pop(0)
                    print(tokenizer.decode([token]), end="")
        print()
    except KeyboardInterrupt:
        print(" press ctrl+D to exit")
    except EOFError:
        print("^D")
        break

