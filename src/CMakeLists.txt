set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#Common libraries to be linked against

#Pick source files
set(SRC_ROOT ${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE STREAMER_SOURCE_FILES ${SRC_ROOT}/*.cpp ${SRC_ROOT}/*.h)
file(GLOB_RECURSE RPC_SOURCE_FILES ${SRC_ROOT}/processor/rpc/*.cpp ${SRC_ROOT}/processor/rpc/*.h)
file(GLOB_RECURSE ZMQ_SOURCE_FILES ${SRC_ROOT}/processor/pubsub/*.cpp ${SRC_ROOT}/pubsub/*.h)
file(GLOB_RECURSE CAFFE_SOURCE_FILES ${SRC_ROOT}/caffe_*.cpp ${SRC_ROOT}/caffe_*.h)
file(GLOB_RECURSE CAFFE_OPENCL_SOURCE_FILES ${SRC_ROOT}/caffe_opencl*.cpp ${SRC_ROOT}/caffe_opencl*.h)
file(GLOB_RECURSE PTGRAY_SOURCE_FILES ${SRC_ROOT}/camera/pgr_*.cpp ${SRC_ROOT}/camera/pgr_*.h)
file(GLOB_RECURSE VIMBA_SOURCE_FILES ${SRC_ROOT}/camera/vimba_*.cpp ${SRC_ROOT}/camera/vimba_*.h)
file(GLOB_RECURSE FRCNN_SOURCE_FILES ${SRC_ROOT}/processor/object_detector.cpp ${SRC_ROOT}/processor/object_detector.h)
file(GLOB_RECURSE NCS_SOURCE_FILES ${SRC_ROOT}/processor/ncs_yolo_detector.cpp ${SRC_ROOT}/processor/ncs_yolo_detector.h)
file(GLOB_RECURSE SSD_SOURCE_FILES ${SRC_ROOT}/processor/ssd_detector.cpp ${SRC_ROOT}/processor/ssd_detector.h)
file(GLOB_RECURSE DLIB_SOURCE_FILES ${PROJECT_SOURCE_DIR}/3rdparty/dlib/dlib/all/source.cpp)
list(REMOVE_ITEM STREAMER_SOURCE_FILES
  ${RPC_SOURCE_FILES}
  ${ZMQ_SOURCE_FILES}
  ${CAFFE_SOURCE_FILES}
  ${CAFFE_OPENCL_SOURCE_FILES}
  ${PTGRAY_SOURCE_FILES}
  ${VIMBA_SOURCE_FILES}
  ${FRCNN_SOURCE_FILES}
  ${NCS_SOURCE_FILES}
  ${SSD_SOURCE_FILES}
  ${DLIB_SOURCE_FILES}
)

# Pick libraries
set(STREAMER_LIBRARIES
  ${GLIB_LIBRARIES}
  ${GLIB_GOBJECT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_APP_LIBRARIES}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
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

if (USE_ZMQ)
  include_directories(SYSTEM ${ZMQ_INCLUDE_DIRS})
  list(APPEND STREAMER_SOURCE_FILES ${ZMQ_SOURCE_FILES})
  list(APPEND STREAMER_LIBRARIES ${ZMQ_LIBRARIES})
endif ()

# Build with FRCNN libraries
if (USE_FRCNN)
  list(APPEND STREAMER_SOURCE_FILES ${FRCNN_SOURCE_FILES})
endif ()

# Build with DLIB libraries
if (USE_DLIB)
  list(APPEND STREAMER_SOURCE_FILES ${DLIB_SOURCE_FILES})
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

list(APPEND STREAMER_LIBRARIES struck)
message(${STREAMER_LIBRARIES})

add_library(streamer SHARED ${STREAMER_SOURCE_FILES})
target_link_libraries(streamer ${STREAMER_LIBRARIES})
ADD_BUILD_REQS(streamer)

if (USE_CAFFE)
  add_caffe_cflags(streamer)
endif ()
