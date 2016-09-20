#!/usr/bin/env bash
sudo apt-get -qq update
sudo apt-get -y install gstreamer1.0 cmake libglib2.0-dev libgoogle-glog-dev libboost-all-dev
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
if [ ! -d "$HOME/installed" ]; then
    mkdir -p $HOME/installed
fi
if [ ! -d "$HOME/installed/Caffe-${CAFFE_COMMIT_HASH}" ]; then
    bash $SCRIPT_DIR/install_caffe_mac.sh
    touch $HOME/installed/Caffe-${CAFFE_COMMIT_HASH}
fi