# Dockerfile for building Streamer with Tensorflow
#
#   Author: Ke-Jou Hsu <nosus_hsu@gatech.edu>

ARG DEFAULT_WORKDIR=/vcs
#FROM streamer_env:tf_v0.5
FROM streamer_env:1-5-1-devel-gpu

COPY ./ $DEFAULT_WORKDIR/streamer

WORKDIR $DEFAULT_WORKDIR/streamer

ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/include

RUN mkdir build 
#&& \ 
#    cd build && \
#    cmake -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda -DUSE_TENSORFLOW=yes -DTENSORFLOW_HOME=/usr/local/include/tensorflow/ .. && \
#    make -j8 && make apps -j8

#WORKDIR $DEFAULT_WORKDIR/streamer/build

