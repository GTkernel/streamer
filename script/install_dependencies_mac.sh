#!/usr/bin/env bash
brew update
brew install cmake glog glib gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly \
    boost opencv

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
bash $SCRIPT_DIR/install_caffe_mac.sh
