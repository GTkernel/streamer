#!/usr/bin/env bash
# Steps to compile Caffe on Mac
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
brew install -vd snappy leveldb gflags glog szip lmdb
brew tap homebrew/science
brew install hdf5 opencv
brew install protobuf boost
brew install openblas
brew install python
brew install numpy

git clone https://github.com/BVLC/caffe
cd caffe
git reset --hard 25422de79f58e214e55834524bfe696f8651889f
patch Makefile.config.example<$SCRIPT_DIR/Caffe_Makefile.config.diff -o Makefile.config
make -j4
sudo make distribute
cd ..