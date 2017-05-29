#!/usr/bin/env bash

# Install dependencies for Caffe
sudo apt-get -y install build-essential cmake git pkg-config libprotobuf-dev \
     libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler \
     libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev python-pip \
     python-dev python-numpy python-scipy
sudo apt-get -y install --no-install-recommends libboost-all-dev

# Set up the Caffe source code
git clone git@github.com:BVLC/caffe.git
cd caffe
git checkout 1.0

# Configure the build
cp Makefile.config.example Makefile.config
echo "OPENCV_VERSION := 3" >> Makefile.config
echo "CPU_ONLY := 1" >> Makefile.config

# Compile Caffe
make all
make test
make runtest
make distribute

cd ..
