#!/usr/bin/env bash
brew install -vd snappy leveldb gflags glog szip lmdb
brew tap homebrew/science
brew install hdf5 opencv
brew install protobuf boost
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release -DCPU_ONLY=ON -DBUILD_docs=OFF -DBUILD_python=OFF -DBUILD_python_layer=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ..
make all
sudo make install