#!/usr/bin/env bash
brew update
brew tap homebrew/science
brew install cmake glog glib gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly \
    boost opencv jemalloc

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Install Caffe
INSTALL_DIR=$HOME/installed_${TRAVIS_OS_NAME}
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
fi

bash $SCRIPT_DIR/install_caffe_mac.sh