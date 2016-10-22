set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# Common libraries to be linked against

# Pick source files
set(SRC_ROOT ${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE TX1DNN_SOURCE_FILES ${SRC_ROOT}/*.cpp ${SRC_ROOT}/*.h)
file(GLOB_RECURSE CAFFE_SOURCE_FILES ${SRC_ROOT}/caffe_*.cpp ${SRC_ROOT}/caffe_*.h)
file(GLOB_RECURSE CAFFE_FP16_SOURCE_FILES ${SRC_ROOT}/caffe_fp16*.cpp ${SRC_ROOT}/caffe_fp16*.h)
file(GLOB_RECURSE CAFFE_OPENCL_SOURCE_FILES ${SRC_ROOT}/caffe_opencl*.cpp ${SRC_ROOT}/caffe_opencl*.h)
list(REMOVE_ITEM CAFFE_SOURCE_FILES ${CAFFE_FP16_SOURCE_FILES} ${CAFFE_OPENCL_SOURCE_FILES})
file(GLOB_RECURSE MXNET_SOURCE_FILES ${SRC_ROOT}/mxnet_*.cpp ${SRC_ROOT}/mxnet_*.h)
file(GLOB_RECURSE GIE_SOURCE_FILES ${SRC_ROOT}/gie_*.cpp ${SRC_ROOT}/gie_*.h)
list(REMOVE_ITEM TX1DNN_SOURCE_FILES
  ${CAFFE_SOURCE_FILES}
  ${CAFFE_FP16_SOURCE_FILES}
  ${CAFFE_OPENCL_SOURCE_FILES}
  ${MXNET_SOURCE_FILES}
  ${GIE_SOURCE_FILES})

# Pick libraries
set(TX1DNN_LIBRARIES
  ${GLIB_LIBRARIES}
  ${GLIB_GOBJECT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_APP_LIBRARIES}
  ${Boost_SYSTEM_LIBRARY})

# Build caffe libraries
if (USE_CAFFE)
  if (USE_FP16)
    list(APPEND TX1DNN_SOURCE_FILES ${CAFFE_FP16_SOURCE_FILES})
  else ()
    list(APPEND TX1DNN_SOURCE_FILES ${CAFFE_SOURCE_FILES})
  endif ()

  list(APPEND TX1DNN_LIBRARIES ${Caffe_LIBRARIES} ${HDF5_LIBRARIES})

  if (USE_CUDA AND Cnmem_FOUND)
    list(APPEND TX1DNN_LIBRARIES ${Cnmem_LIBRARIES})
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

# Build GIE libraries
if (USE_GIE)
  list(APPEND TX1DNN_SOURCE_FILES ${GIE_SOURCE_FILES})
  list(APPEND TX1DNN_LIBRARIES ${GIE_LIBRARIES})
endif ()

# Build MXNet libraries
if (USE_MXNET)
  list(APPEND TX1DNN_SOURCE_FILES ${MXNET_SOURCE_FILES})
  list(APPEND TX1DNN_LIBRARIES ${MXNet_LIBRARIES})
endif ()

add_library(tx1dnn SHARED ${TX1DNN_SOURCE_FILES} utils/math_utils.h)
target_link_libraries(tx1dnn ${TX1DNN_LIBRARIES})
if (USE_CAFFE)
  add_caffe_cflags(tx1dnn)
endif ()