#!/bin/bash

BAZEL_VERSION=0.8.0
#downgrade bazel
curl -LO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb" 
dpkg -i bazel_*.deb

#install tensorflow
cd /vcs
git clone https://github.com/tensorflow/tensorflow.git

cd ./tensorflow
git fetch
git checkout r1.5
echo -ne '\n\n\n\n\n\n\n\n\n\n\n\n\n\n' | ./configure
bazel build --config=monolithic --config=opt //tensorflow:libtensorflow_cc.so
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
mkdir -p /tmp/tensorflow_pkg
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.5.1-cp27-cp27mu-linux_x86_64.whl
cp -r /usr/local/lib/python2.7/dist-packages/tensorflow/include/* /usr/local/include/
cp -r tensorflow/cc /usr/local/include/tensorflow/
cp bazel-genfiles/tensorflow/cc/ops/*.h /usr/local/include/tensorflow/cc/ops/
cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/

cd /vcs
rm -rf /tmp/tensorflow_pkg
rm -rf ./tensorflow

