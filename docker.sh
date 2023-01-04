#!/bin/sh
docker build -t ubuntu2004_torch1121_cuda113 .

docker run -it --name ctci --gpus all --shm-size 120G \
    -v /home/parzival/Downloads/CTCI/:/app/ \
    -p 5002:5002 \
    --net=host --env="DISPLAY" \
    --volume= "$HOME/.Xauthority:/root/.Xauthority" \
    ubuntu2004_torch1121_cuda113 bash