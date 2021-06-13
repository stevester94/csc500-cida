#FROM ubuntu:20.04
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install easydict 
RUN pip3 install numpy 
RUN pip3 install torch
RUN pip3 install torch  --upgrade
RUN pip3 install numpy  --upgrade
RUN pip3 install matplotlib
RUN pip3 install sklearn

ADD requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /csc500-super-repo/csc500-cida
