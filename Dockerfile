FROM python:3.8.8

USER root

WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Europe/London
ENV HOME=/config
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
