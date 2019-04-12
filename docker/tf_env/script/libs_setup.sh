#!/bin/bash
cd /vcs

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
BAZEL_VERSION=0.15.0
#downgrade bazel
curl -LO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb"
dpkg -i bazel_*.deb
rm -f bazel_*.deb

# new protobuf installation
#based on tensorflow r1.12 requirement
wget https://github.com/google/protobuf/archive/v3.6.0.tar.gz
tar xvf v3.6.0.tar.gz
rm v3.6.0.tar.gz
cd protobuf-3.6.0/
./autogen.sh
./configure
make -j`nproc` && make install
rm -rf /vcs/protobuf-3.6.0

# install tensorflow
cd /vcs
wget https://github.com/tensorflow/tensorflow/archive/v1.12.0.tar.gz
tar xvf v1.12.0.tar.gz
rm v1.12.0.tar.gz
cd tensorflow-1.12.0

pip install enum34 mock
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
pip install h5py==2.8.0

echo -ne '\n\n\n\n\n\n\n\n\n\n\n' | ./configure
bazel build --config=monolithic //tensorflow:libtensorflow_cc.so
bazel build //tensorflow/tools/pip_package:build_pip_package

mkdir -p /tmp/tensorflow_pkg
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.12.0-cp27-cp27mu-linux_x86_64.whl
cp -r /usr/local/lib/python2.7/dist-packages/tensorflow/include/* /usr/local/include/
cp -r tensorflow/cc /usr/local/include/tensorflow/
cp bazel-genfiles/tensorflow/cc/ops/*.h /usr/local/include/tensorflow/cc/ops/
cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/

rm -rf /tmp/tensorflow_pkg
rm -rf /vcs/tensorflow-1.12.0

#install grpc
cd /vcs
git clone -b v1.19.0 https://github.com/grpc/grpc.git
cd grpc
git submodule update --init
sed -i "s/ldconfig/ldconfig -r \/usr\/local\/bin\//g" Makefile
make -j8 && make install

rm -rf /vcs/grpc

