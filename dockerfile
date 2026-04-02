#FROM nvcr.io/nvidia/pytorch:24.01-py3
#FROM vastai/pytorch
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends git build-essential python3 python3-pip python3-dev wget
# && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel psutil


RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb &&\
#    dpkg -i cuda-keyring_1.1-1_all.deb &&\
#    apt-get update &&\
#    apt-get -y install cuda-toolkit-13-0


#ENV TORCH_CUDA_ARCH_LIST="9.0"
#ENV CUDA_HOME=/usr/local/cuda
    
RUN pip install ninja
RUN pip install flash-attn --no-build-isolation
#RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl &&\
#    pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
#





WORKDIR /llm

COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["/bin/bash"]
ENTRYPOINT []