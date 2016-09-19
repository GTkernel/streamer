# Caffe package for CNN Triplet training
unset(Caffe_FOUND)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/vision_layers.hpp
  HINTS
  ${CAFFE_HOME}/include
  /usr/local/include)

find_library(Caffe_LIBRARIES NAMES caffe-nv caffe
  HINTS
  ${CAFFE_HOME}/lib
  /usr/local/lib)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Caffe DEFAULT_MSG
  Caffe_INCLUDE_DIRS Caffe_LIBRARIES)

if (Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
  set(Caffe_FOUND 1)
else ()
  message("Not found Caffe")
endif ()
