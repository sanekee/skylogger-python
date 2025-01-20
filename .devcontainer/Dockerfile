FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt install -y \
    build-essential \
    wget \
    unzip \
    bc \
    vim \
    libleptonica-dev \
    git \
    make \
    g++ \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libpng-dev \
    libjpeg8-dev \
    libtiff5-dev libicu-dev \
    libpango1.0-dev \
    autoconf-archive  \
    python3-pip \
    python3-numpy \
    python3-skimage \
    python3-pil \
    python3-scipy \
    python3-toolz \
    opencv-data \
    tesseract-ocr \
    tesseract-ocr-all \
    ffmpeg


# use pip for intellisense
#python3-opencv

FROM base as runtime

ARG USERNAME=dev
ARG VCS_SERVER_PATH="/home/${USERNAME}/.vscode-server"

RUN groupadd --gid 1000 ${USERNAME}
RUN useradd -rm -d /home/${USERNAME} -s /bin/bash --gid 1000 -G sudo --uid 1000 ${USERNAME}

VOLUME [ "${VCS_SERVER_PATH}"]

RUN set -eux; \
    mkdir -p ${VCS_SERVER_PATH} && \
    chown ${USERNAME}:${USERNAME} ${VCS_SERVER_PATH}

USER 1000:1000
ENV HOME=

WORKDIR /app

COPY requirement.txt .

RUN pip3 install -r requirement.txt

ENTRYPOINT [ "tail", "-f", "/dev/null" ]
