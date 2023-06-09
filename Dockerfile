FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

ADD torchsig/ /build/torchsig

ADD pyproject.toml /build/pyproject.toml

RUN pip3 install /build

RUN pip3 install notebook

WORKDIR /workspace/code

ADD examples/ /workspace/code/examples