#!/usr/bin/env bash
sudo apt-get -y install gstreamer1.0 cmake libglib2.0-dev libgoogle-glog-dev libboost-all-dev libopencv-dev
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
INSTALL_DIR=$HOME/installed_${TRAVIS_OS_NAME}
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
fi
if [ ! -f "$INSTALL_DIR/Caffe-${CAFFE_COMMIT_HASH}" ]; then
    bash $SCRIPT_DIR/install_caffe_linux.sh
    # install successful, add a mark
    if [ $? -eq 0 ]; then
        touch $INSTALL_DIR/Caffe-${CAFFE_COMMIT_HASH}
    fi
fi