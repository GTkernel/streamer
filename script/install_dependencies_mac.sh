#!/usr/bin/env bash
brew update
brew install cmake glog glib gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg \
    boost opencv
bash install_caffe_mac.sh
