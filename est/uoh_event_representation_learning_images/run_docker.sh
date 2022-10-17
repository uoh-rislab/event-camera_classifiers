#!/bin/bash

docker rm -fv ignacio_erl_affwild100ms

nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0  --name ignacio_erl_affwild100ms -v /home/ignacio.bugueno/cachefs/erl/input:/app/input -v /home/ignacio.bugueno/cachefs/erl/output:/app/output -v /home/ignacio.bugueno/cachefs/erl/log:/app/log ignacio_erl_affwild100ms

#sleep 10; docker logs ignacio_erl
