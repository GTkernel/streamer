#!/usr/bin/env bash
# Install Caffe
sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get -y install --no-install-recommends libboost-all-dev
sudo apt-get -y install libgflags-dev libgoogle-glog-dev
sudo apt-get -y install libatlas-base-dev
git clone https://github.com/BVLC/caffe
cd caffe
git reset --hard 25422de79f58e214e55834524bfe696f8651889f
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release -DCPU_ONLY=ON -DBUILD_docs=OFF -DBUILD_python=OFF -DBUILD_python_layer=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DUSE_LMDB=off ..
make -j`nproc`
sudo make install
cd ..
cd ..