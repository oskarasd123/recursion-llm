# recursion-llm
This is a project where I study llm-s and try to train one.<br>
I have taken a lot of inspiration from [modded nanogpt](https://github.com/KellerJordan/modded-nanogpt).<br>
My goal is to make a somewhat optimised training setup and then try to find modifications that improve the model's learning efficiency.<br>
## If you want to run it yourself
- Install dependencies with commands in `setup.sh`(This script is mostly for automated installation in containers).
- Download the dataset. This happens on first run of either `dataloader.py` or `train.py`.
- Run training script with `python3 train.py` or `torchrun --nproc_per_node=2 train.py` when training with multiple gpus.
NB! running train.py on multiple gpus before downloading the dataset will start multiple downloads.