# Common libraries to be linked against
set(TX1DNN_LIBRARIES
  ${GLIB_LIBRARIES}
  ${GLIB_GOBJECT_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_APP_LIBRARIES}
  ${Boost_SYSTEM_LIBRARY}
  ${JSONCPP_LIBRARIES})

set(TX1DNN_SOURCE_FILES
  video/gst_video_capture.cpp
  video/gst_video_capture.h
  common/utils.h
  common/data_buffer.h
  common/types.cpp
  classifier/classifier.cpp
  classifier/classifier.h)

# Build caffe libraries
if (USE_CAFFE)
  if (USE_FP16)
    set(CAFFE_SOURCE_FILES classifier/caffe_fp16_classifier.h classifier/caffe_fp16_classifier.cpp)
  else ()
    set(CAFFE_SOURCE_FILES classifier/caffe_classifier.cpp classifier/caffe_classifier.h)
  endif ()

  list(APPEND TX1DNN_SOURCE_FILES ${CAFFE_SOURCE_FILES})
  list(APPEND TX1DNN_LIBRARIES ${Caffe_LIBRARIES} ${HDF5_LIBRARIES})

  if (Cnmem_FOUND)
    list(APPEND TX1DNN_LIBRARIES ${Cnmem_LIBRARIES})
  endif ()

  # target_include_directories(tx1dnn-caffe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif ()
# Set Caffe compile flags
function(add_caffe_cflags TARGET)
  SET(CAFFE_CFLAGS "-MMD -MP -pthread -fPIC -O2 -DUSE_OPENCV")

  if (TEGRA)
    SET(CAFFE_CFLAGS "${CAFFE_CFLAGS} -DUSE_CNMEN -DUSE_CUDNN")
  endif ()

  get_target_property(TEMP ${TARGET} COMPILE_FLAGS)
  if (TEMP STREQUAL "TEMP-NOTFOUND")
    SET(TEMP "") # set to empty string
  endif ()
  # append our values
  SET(TEMP "${CAFFE_CFLAGS} ${TEMP}")
  set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TEMP})
endfunction()

# add_caffe_cflags(tx1dnn-caffe)

# Build GIE libraries
if (USE_GIE)
  set(GIE_SOURCE_FILES
    classifier/gie_classifier.cpp
    classifier/gie_classifier.h
    classifier/gie_inferer.cpp
    classifier/gie_inferer.h)

  list(APPEND TX1DNN_SOURCE_FILES ${GIE_SOURCE_FILES})
  list(APPEND TX1DNN_LIBRARIES ${GIE_LIBRARIES})
endif ()

# Build MXNet libraries
if (USE_MXNET)
  set(MXNET_SOURCE_FILES classifier/mxnet_classifier.cpp classifier/mxnet_classifier.h ${COMMON_SOURCE_FILES})

  list(APPEND TX1DNN_SOURCE_FILES ${MXNET_SOURCE_FILES})
  list(APPEND TX1DNN_LIBRARIES ${MXNet_LIBRARIES})
endif ()

add_library(tx1dnn SHARED ${TX1DNN_SOURCE_FILES})
target_link_libraries(tx1dnn ${TX1DNN_LIBRARIES})
if (USE_CAFFE)
  add_caffe_cflags(tx1dnn)
endif ()
