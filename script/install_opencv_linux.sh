#!/usr/bin/env bash

# Install dependencies for OpenCV
sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config \
     libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
     libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev \
     libdc1394-22-dev

# Set up the OpenCV test data
git clone https://github.com/opencv/opencv_extra.git
export OPENCV_TEST_DATA_PATH=$(pwd)/opencv_extra/testdata

# Set up the OpenCV source code
git clone git@github.com:opencv/opencv.git
cd opencv
git checkout 3.2.0

# Compile OpenCV
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install

# Run basic tests
./bin/opencv_test_core

cd ../..
