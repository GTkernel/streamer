#!/bin/bash
cd /vcs
git clone https://github.com/opencv/opencv.git
cd opencv 
git reset --hard d34eec3ab3940c085b12d0420d9bbe76ea25e620

#sed -i "6i #include <cuda_fp16.h>" modules/cudev/include/opencv2/cudev/common.hpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DWITH_CUDA=OFF \
      -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_EIGEN=ON \
      -DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF ..

#      -DWITH_CUDA=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" \
make -j16 && make install
cd ../.. 
rm -rf opencv
