
# used for faster container setup
apt update
apt install -y --no-install-recommends git build-essential
pip install --upgrade pip setuptools wheel
# flash_attn needs torch with cuda version 12.1 to install.
# afterwards you can upgrade to cuda 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

#pip install ninja
#pip install packaging
pip install flash-attn --no-build-isolation # takes a long time
pip install -r requirements.txt