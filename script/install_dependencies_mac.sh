#!/usr/bin/env bash
brew update
brew tap homebrew/science
brew install cmake glog glib gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly \
    boost opencv

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
INSTALL_DIR=$HOME/installed_${TRAVIS_OS_NAME}
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
fi
if [ ! -f "$INSTALL_DIR/Caffe-${CAFFE_COMMIT_HASH}-mac" ]; then
    bash $SCRIPT_DIR/install_caffe_mac.sh
    # install successful, add a mark
    if [ $? -eq 0 ]; then
        touch $INSTALL_DIR/Caffe-${CAFFE_COMMIT_HASH}-mac
    fi
fi