#! /bin/bash
#-u $(id -u):$(id -g) \
docker run \
--gpus=all \
--network=host  \
-ti \
--rm  \
-v $CSC500_ROOT_PATH:/csc500-super-repo  \
--shm-size=16g \
csc500-cida-torch
