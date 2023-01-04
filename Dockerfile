FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

RUN apt-get -y update
RUN apt-get upgrade -y
RUN apt-get -y install python3
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y cmake
RUN apt-get -y install python3-pip
RUN apt-get -y update && apt-get install -y libopencv-dev
RUN apt-get install -y git
RUN apt-get install -y tmux
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get clean
WORKDIR /app