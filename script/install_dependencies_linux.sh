#!/usr/bin/env bash
sudo apt-get -qq update
sudo apt-get -y install gstreamer1.0 cmake libglib2.0-dev libgoogle-glog-dev libboost-all-dev
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install OpenCV
bash $SCRIPT_DIR/install_opencv_linux.sh
echo "OpenCV installed"
# Install Caffe
bash $SCRIPT_DIR/install_caffe_linux.sh