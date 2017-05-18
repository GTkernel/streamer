#!/usr/bin/env bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libopencv-dev gstreamer1.0 libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev libjemalloc-dev libzmq3-dev \
    libeigen3-dev gcc-5 g++-5

# The nlohmann::json library does not work with gcc/g++ 4.8, so we will use
# version 5 instead.
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 \
     --slave /usr/bin/g++ g++ /usr/bin/g++-5

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
