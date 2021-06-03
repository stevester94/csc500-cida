#! /bin/bash
#-u $(id -u):$(id -g) \
docker run \
--network=host  \
-ti \
--rm  \
-v $CSC500_ROOT_PATH:/csc500-super-repo  \
csc500-cida-torch
