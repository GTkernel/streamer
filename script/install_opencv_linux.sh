#!/usr/bin/env bash
sudo apt-get -y install build-essential
sudo apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
wget https://github.com/opencv/opencv/archive/2.4.13.tar.gz
tar xvf 2.4.13.tar.gz
cd opencv-2.4.13
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j`nproc`
sudo make install
cd ..
cd ..