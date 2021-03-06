# Copyright 2016 The Streamer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#Common libraries to be linked against

#Pick source files
set(SRC_ROOT ${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE STREAMER_SOURCE_FILES
  ${SRC_ROOT}/*.cpp
  ${SRC_ROOT}/*.h
)
file(GLOB_RECURSE RPC_SOURCE_FILES
  ${SRC_ROOT}/processor/rpc/*.cpp
  ${SRC_ROOT}/processor/rpc/*.h
)
file(GLOB_RECURSE CAFFE_SOURCE_FILES
  ${SRC_ROOT}/caffe_*.cpp
  ${SRC_ROOT}/caffe_*.h
)
file(GLOB_RECURSE TENSORFLOW_SOURCE_FILES
  ${SRC_ROOT}/tf_*.cpp
  ${SRC_ROOT}/tf_*.h
)
file(GLOB_RECURSE CAFFE_OPENCL_SOURCE_FILES
  ${SRC_ROOT}/caffe_opencl*.cpp
  ${SRC_ROOT}/caffe_opencl*.h
)
file(GLOB_RECURSE PTGRAY_SOURCE_FILES
  ${SRC_ROOT}/camera/pgr_*.cpp
  ${SRC_ROOT}/camera/pgr_*.h
)
file(GLOB_RECURSE VIMBA_SOURCE_FILES
  ${SRC_ROOT}/camera/vimba_*.cpp
  ${SRC_ROOT}/camera/vimba_*.h
)
file(GLOB_RECURSE FRCNN_SOURCE_FILES
  ${SRC_ROOT}/frcnn_*.cpp
  ${SRC_ROOT}/frcnn_*.h
)
file(GLOB_RECURSE NCS_SOURCE_FILES
  ${SRC_ROOT}/ncs.h
  ${SRC_ROOT}/ncs_*.cpp
  ${SRC_ROOT}/ncs_*.h
)
file(GLOB_RECURSE SSD_SOURCE_FILES
  ${SRC_ROOT}/ssd_*.cpp
  ${SRC_ROOT}/ssd_*.h
)
file(GLOB_RECURSE DLIB_SOURCE_FILES
  ${SRC_ROOT}/dlib_*.cpp
  ${SRC_ROOT}/dlib_*.h
)
file(GLOB_RECURSE IMAGEMATCH_SOURCE_FILES
  ${SRC_ROOT}/processor/imagematch/*.cpp
  ${SRC_ROOT}/processor/imagematch/*.h
)
list(REMOVE_ITEM STREAMER_SOURCE_FILES
  ${RPC_SOURCE_FILES}
  ${CAFFE_SOURCE_FILES}
  ${TENSORFLOW_SOURCE_FILES}
  ${CAFFE_OPENCL_SOURCE_FILES}
  ${PTGRAY_SOURCE_FILES}
  ${VIMBA_SOURCE_FILES}
  ${FRCNN_SOURCE_FILES}
  ${NCS_SOURCE_FILES}
  ${SSD_SOURCE_FILES}
  ${DLIB_SOURCE_FILES}
  ${IMAGEMATCH_SOURCE_FILES}
)

# Pick libraries
set(STREAMER_LIBRARIES
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${ZMQ_LIBRARIES}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
)

# Build caffe libraries
if (USE_CAFFE)
  list(APPEND STREAMER_SOURCE_FILES ${CAFFE_SOURCE_FILES})

  list(APPEND STREAMER_LIBRARIES ${Caffe_LIBRARIES})

  if (USE_CUDA AND Cnmem_FOUND)
    list(APPEND STREAMER_LIBRARIES ${Cnmem_LIBRARIES})
  endif ()
endif ()

# Build TensorFlow libraries
if (USE_TENSORFLOW)
  list(APPEND STREAMER_SOURCE_FILES ${TENSORFLOW_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES ${TensorFlow_LIBRARIES})
endif ()

# Set Caffe compile flags
function(add_caffe_cflags TARGET)
  SET(CAFFE_CFLAGS "-MMD -MP -pthread -fPIC -DUSE_OPENCV")

  if (USE_CUDA)
    SET(CAFFE_CFLAGS "${CAFFE_CFLAGS} -DUSE_CUDNN")
    if (Cnmem_FOUND)
      SET(CAFFE_CFLAGS "${CAFFE_CFLAGS} -DUSE_CNMEM")
    endif ()
  elseif (USE_OPENCL)
    # Put any flag needed to build with opencl here
  endif ()

  get_target_property(TEMP ${TARGET} COMPILE_FLAGS)
  if (TEMP STREQUAL "TEMP-NOTFOUND")
    SET(TEMP "") # set to empty string
  endif ()
  # append our values
  SET(TEMP "${CAFFE_CFLAGS} ${TEMP}")
  set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TEMP})
endfunction()

# Build with PtGray SDK
if (USE_PTGRAY)
  list(APPEND STREAMER_SOURCE_FILES ${PTGRAY_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES ${PtGray_LIBRARIES})
endif ()

# Build with VIMBA SDK
if (USE_VIMBA)
  list(APPEND STREAMER_SOURCE_FILES ${VIMBA_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES ${Vimba_LIBRARIES})
endif ()

# Build RPC support
if (USE_RPC)
  file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/processor/rpc/*.proto")
  PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
  PROTOBUF_GENERATE_GRPC_CPP(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR})
  # Silence compiler warnings in automatically generated source files
  foreach (f ${PROTO_SRCS} ${GRPC_SRCS})
    get_filename_component(ff ${f} NAME)
    set_source_files_properties(${ff} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endforeach ()
  list(APPEND STREAMER_SOURCE_FILES ${RPC_SOURCE_FILES})
  list(APPEND STREAMER_SOURCE_FILES ${PROTO_SRCS} ${PROTO_HDRS})
  list(APPEND STREAMER_SOURCE_FILES ${GRPC_SRCS} ${GRPC_HDRS})
  list(APPEND STREAMER_LIBRARIES ${PROTOBUF_LIBRARIES})
  list(APPEND STREAMER_LIBRARIES ${GRPC_LIBRARIES})
endif ()

# Build ImageMatch
if (USE_TENSORFLOW)
  list(APPEND STREAMER_SOURCE_FILES ${IMAGEMATCH_SOURCE_FILES})
endif()

# Build with FRCNN libraries
if (USE_FRCNN)
  list(APPEND STREAMER_SOURCE_FILES ${FRCNN_SOURCE_FILES})
endif ()

# Build with DLIB libraries
if (USE_DLIB)
  list(APPEND STREAMER_SOURCE_FILES ${DLIB_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES dlib)
endif ()

# Build with NCS libraries
if (USE_NCS)
  list(APPEND STREAMER_SOURCE_FILES ${NCS_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES ${NCS_LIBRARIES})
endif ()

# Build with ATHENA libraries
if (USE_ATHENA)
  list(APPEND STREAMER_LIBRARIES ${Athena_LIBRARIES} ${Athena_utils_LIBRARIES})
endif ()

# Build with SSD libraries
if (USE_SSD)
  list(APPEND STREAMER_SOURCE_FILES ${SSD_SOURCE_FILES})
endif ()

message(${STREAMER_LIBRARIES})

add_library(streamer SHARED ${STREAMER_SOURCE_FILES})
target_link_libraries(streamer ${STREAMER_LIBRARIES})
ADD_BUILD_REQS(streamer)

if (USE_CAFFE)
  add_caffe_cflags(streamer)
endif ()
