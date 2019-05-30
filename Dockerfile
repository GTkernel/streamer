# Dockerfile for building Streamer with Tensorflow
#
#   Author: Ke-Jou Hsu <nosus_hsu@gatech.edu>

#For gpu
#FROM streamer_env:cuda9-tf1.12
#For cpu
FROM streamer_env:tf1.12

ARG DEFAULT_WORKDIR=/vcs
COPY ./ $DEFAULT_WORKDIR/streamer

WORKDIR $DEFAULT_WORKDIR/streamer

ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/include

RUN mkdir build && \ 
    cd build && \
# For gpu
#    cmake -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda -DUSE_TENSORFLOW=yes -DTENSORFLOW_HOME=/usr/local/include/tensorflow/ -DUSE_RPC=yes .. && \
# For cpu
    cmake -DCMAKE_BUILD_TYPE=Release -DBACKEND=cpu -DUSE_TENSORFLOW=yes -DTENSORFLOW_HOME=/usr/local/include/tensorflow/ -DUSE_RPC=yes .. && \
    make -j8 && make apps -j8

RUN apt-get -y autoremove && apt-get autoclean

WORKDIR $DEFAULT_WORKDIR/streamer/build

