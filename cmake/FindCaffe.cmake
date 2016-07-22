# Caffe package for CNN Triplet training
unset(CAFFE_FOUND)

find_path(CAFFE_INCLUDE_DIR NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/vision_layers.hpp
  HINTS
  /usr/local/include)

find_library(CAFFE_LIBS NAMES caffe
  HINTS
  /usr/local/lib)

if(CAFFE_LIBS AND CAFFE_INCLUDE_DIR)
    set(CAFFE_FOUND 1)
endif()