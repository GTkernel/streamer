#!/usr/bin/env bash
sudo apt-get -qq update
sudo apt-get -y install sudo apt-get install -y gstreamer1.0 cmake libglib2.0-dev libgoogle-glog-dev libboost-all-dev
# Install OpenCV
bash install_opencv_linux.sh
echo "OpenCV installed"
# Install Caffe
bash install_caffe_linux.sh