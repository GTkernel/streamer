#!/bin/bash
cd /vcs

#ProtoBuf
git clone https://github.com/google/protobuf.git
cd protobuf
git reset --hard 072431452a365450c607e9503f51786be44ecf7f
./autogen.sh
./configure
make -j`nproc`
make install 
cd .. 
rm -rf protobuf

#OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv 
git reset --hard d34eec3ab3940c085b12d0420d9bbe76ea25e620
    
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DWITH_CUDA=OFF \
      -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_EIGEN=ON \
      -DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF ..

make -j`nproc` && make install
cd ../.. 
rm -rf opencv

#install bazel
rm /etc/apt/apt.conf.d/docker-clean
apt-get update
apt-get install -y bash-completion
BAZEL_VERSION=0.10.0
#downgrade bazel
curl -LO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb"
dpkg -i bazel_*.deb
rm -f bazel_*.deb

#tensorflow CUDA requirement
pip install enum34 mock

# install tensorflow
cd /vcs
wget https://github.com/tensorflow/tensorflow/archive/v1.8.0.tar.gz
tar xvf v1.8.0.tar.gz
rm v1.8.0.tar.gz
cd tensorflow-1.8.0
echo '\n\n\nn\nn\nn\nn\n\n\n\n\ny\n9.2\n\n\n/usr/lib/x86_64-linux-gnu\n\n\n\n\n\n\n\n\n' | ./configure

bazel build --config=monolithic --config=opt --config=cuda //tensorflow:libtensorflow_cc.so
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
mkdir -p /tmp/tensorflow_pkg
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
cp -r /usr/local/lib/python2.7/dist-packages/tensorflow/include/* /usr/local/include/
cp -r tensorflow/cc /usr/local/include/tensorflow/
cp bazel-genfiles/tensorflow/cc/ops/*.h /usr/local/include/tensorflow/cc/ops/
cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/

cd /vcs
rm -rf /tmp/tensorflow_pkg
rm -rf /tensorflow
