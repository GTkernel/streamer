# Dockerfile for building Streamer's Tensorflow environment
#   
#   Author: Ke-Jou Hsu <nosus_hsu@gatech.edu>

FROM ubuntu:16.04

ARG DEFAULT_WORKDIR=/vcs
# Prepare toolchain
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get --no-install-recommends install -y \
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
	libavcodec-dev:amd64=7:2.8.11-0ubuntu0.16.04.1 \
	libavformat-dev:amd64=7:2.8.11-0ubuntu0.16.04.1 \
	libswscale-dev:amd64=7:2.8.11-0ubuntu0.16.04.1 \
	python-dev=2.7.11-1 \
	python-numpy=1:1.11.0-1ubuntu1 \
	libtbb2:amd64=4.4~20151115-0ubuntu3 \
	libtbb-dev:amd64=4.4~20151115-0ubuntu3 \
	libjpeg-dev:amd64=8c-2ubuntu8 \
	libpng12-dev:amd64=1.2.54-1ubuntu1 \
	libtiff-dev \
	libjasper-dev=1.900.1-debian1-2.4ubuntu1.1 \
	libdc1394-22-dev:amd64=2.2.4-1 \
	libopenexr-dev=2.2.0-10ubuntu2 \
	libeigen3-dev=3.3~beta1-2 \
	libgstreamer1.0-dev=1.8.3-1~ubuntu0.1 \
	libgstreamer-plugins-base1.0-dev=1.8.3-1ubuntu0.2 \
	libgstreamer-plugins-good1.0-dev=1.8.3-1ubuntu0.4 \
	libgstreamer-plugins-bad1.0-dev=1.8.3-1ubuntu0.2 \
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
	openjdk-8-jdk 
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install setuptools && pip install pyyaml && pip install wheel

# Prepare the environment
RUN mkdir -p $DEFAULT_WORKDIR
ENV PATH ${PATH}:/usr/local/bin
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

WORKDIR $DEFAULT_WORKDIR
COPY ./script/libs_setup.sh ./
RUN sh ./libs_setup.sh

# Resolve ProtoBuf version conflict
RUN apt-get update && \
    apt-get -f install -y && \
    apt-get install -y --allow-downgrades libmirprotobuf3:amd64=0.21.0+16.04.20160330-0ubuntu1 && \
	apt-get install -y gstreamer1.0-plugins-bad

