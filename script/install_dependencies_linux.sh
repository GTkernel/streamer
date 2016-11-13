#!/usr/bin/env bash
sudo apt-get install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libopencv-dev gstreamer1.0 libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev libjemalloc-dev libzmq3-dev

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
INSTALL_DIR=$HOME/installed_${TRAVIS_OS_NAME}
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
fi

CAFFE_DIR=$INSTALL_DIR/caffe-${CAFFE_COMMIT_HASH}
if [ -d "$CAFFE_DIR" ] && [ -e "$CAFFE_DIR/build" ]; then
    echo "Using cached Caffe build ($CAFFE_COMMIT_HASH)"
else
    bash $SCRIPT_DIR/install_caffe_linux.sh
fi