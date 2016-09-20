#!/usr/bin/env bash
# Install Caffe
sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get -y install --no-install-recommends libboost-all-dev
sudo apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get instal -y python-pip
sudo pip install numpy
wget https://github.com/BVLC/caffe/archive/rc3.tar.gz
tar xvf rc3.tar.gz
cd caffe-rc3
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release -DCPU_ONLY=ON -DBUILD_docs=OFF -DBUILD_python=OFF -DBUILD_python_layer=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j`nproc`
sudo make install
cd ..
cd ..