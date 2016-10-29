#!/usr/bin/env bash
sudo apt-get install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libopencv-dev gstreamer1.0 libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev libjemalloc-dev

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
INSTALL_DIR=$HOME/installed_${TRAVIS_OS_NAME}
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
fi

bash $SCRIPT_DIR/install_caffe_linux.sh