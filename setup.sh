
# used for faster container setup
apt update
apt install -y --no-install-recommends git build-essential screen
pip install --upgrade pip setuptools wheel
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

pip install flash-attn --no-build-isolation # takes a long time if it builds from source
pip install -r requirements.txt
screen -dmS tensorboard tensorboard --logdir=runs