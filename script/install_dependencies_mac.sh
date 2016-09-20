#!/usr/bin/env bash
brew update
brew install cmake glog glib gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly \
    boost opencv

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
if [ ! -d "$HOME/installed" ]; then
    mkdir -p $HOME/installed
fi
if [ ! -d "$HOME/installed/Caffe-${CAFFE_COMMIT_HASH}" ]; then
    bash $SCRIPT_DIR/install_caffe_mac.sh
    touch $HOME/installed/Caffe-${CAFFE_COMMIT_HASH}
fi
