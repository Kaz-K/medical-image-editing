FROM nvcr.io/nvidia/pytorch:20.01-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-venv \
    python3-distutils \
    vim less \
    zip \
    unzip \
    git \
    python3-tk \
    libpython3.8-dev

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip3 install -U pip setuptools

RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

ARG DOCKER_UID=1000
ARG DOCKER_USER=docker
ARG DOCKER_PASSWORD=docker
RUN useradd -m \
    --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
    && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

RUN mkdir -p /${DOCKER_USER}

ENV PYTHONPATH /${DOCKER_USER}/src

COPY requirements.txt /${DOCKER_USER}

RUN pip3 install -r /${DOCKER_USER}/requirements.txt

USER ${DOCKER_USER}
