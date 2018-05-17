#!/bin/bash
cd /vcs
git clone https://github.com/opencv/opencv.git
cd opencv 
git reset --hard d34eec3ab3940c085b12d0420d9bbe76ea25e620
    
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_EIGEN=ON \
      -DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF ..

make -j`nproc` && make install
cd ../.. 
rm -rf opencv
