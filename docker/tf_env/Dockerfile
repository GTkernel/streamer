# Dockerfile for building Streamer's Tensorflow environment
#   
#   Author: Ke-Jou Hsu <nosus_hsu@gatech.edu>


FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ARG DEFAULT_WORKDIR=/vcs
# Prepare toolchain
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get --no-install-recommends install -y -q \
	ssh \
	sudo \
	rsync \
	wget=1.17.1-1ubuntu1.* \
	cmake=3.5.1-1ubuntu3 \
	git=1:2.7.4-0ubuntu1.* \
	vim=2:7.4.1689-3ubuntu1.2 \
	autoconf=2.69-9 \
	automake=1:1.15-4ubuntu1 \
	libtool=2.4.6-0.1 \
	curl>7.47.0-1ubuntu2.2 \
	g++=4:5.3.1-1ubuntu1 \
	unzip=6.0-20ubuntu1 \
	build-essential=12.1ubuntu2 \
	ocl-icd-opencl-dev:amd64=2.2.8-1 \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	python-dev \
	python-numpy \
	libtbb2:amd64 \
	libtbb-dev:amd64 \
	libjpeg-dev:amd64 \
	libpng12-dev:amd64 \
	libtiff-dev \
	libjasper-dev \
	libdc1394-22-dev:amd64=2.2.4-1 \
	libopenexr-dev=2.2.0-10ubuntu2 \
	libeigen3-dev=3.3~beta1-2 \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer-plugins-good1.0-dev \
	libgstreamer-plugins-bad1.0-dev \
	python3-dev=3.5.1-3 \
	libleveldb-dev:amd64=1.18-5 \
	libsnappy-dev:amd64=1.1.3-2 \
	libhdf5-serial-dev=1.8.16+docs-4ubuntu1 \
	libboost-all-dev=1.58.0.1ubuntu1 \
	libgflags-dev=2.1.2-3 \
	libgoogle-glog-dev=0.3.4-0.1 \
	liblmdb-dev:amd64=0.9.17-3 \
	python-scipy=0.17.0-1 \
	gstreamer1.0:amd64 \
	libjemalloc-dev=3.6.0-9ubuntu1 \	
	libzmq3-dev:amd64=4.1.4-7 \
	libeigen3-dev=3.3~beta1-2 \
	libblas-dev=3.6.0-2ubuntu2 \
	libgtk2.0-dev=2.24.30-1ubuntu1.16.04.2 \
	python-pip=8.1.1-2ubuntu0.4 \
	openjdk-8-jdk && \
	rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install setuptools && \
    python -m pip install pyyaml && \
    python -m pip install wheel

# Prepare the environment
RUN mkdir -p $DEFAULT_WORKDIR
ENV PATH ${PATH}:/usr/local/bin
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

WORKDIR $DEFAULT_WORKDIR
COPY ./script/libs_setup_gpu.sh ./
RUN sh ./libs_setup_gpu.sh

# Resolve ProtoBuf version conflict
RUN apt-get update && \
    apt-get -f install -y && \
    apt-get install -y --allow-downgrades libmirprotobuf3:amd64=0.21.0+16.04.20160330-0ubuntu1 && \
	apt-get install -y gstreamer1.0-plugins-bad

