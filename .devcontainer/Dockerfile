FROM ubuntu:22.04

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

WORKDIR /app

COPY requirement.txt .

RUN pip3 install -r requirement.txt

ENTRYPOINT [ "tail", "-f", "/dev/null" ]
