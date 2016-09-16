set(COMMON_LIBRARIES
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${Boost_SYSTEM_LIBRARY})

set(COMMON_SOURCE_FILES
        video/gst_video_capture.cpp
        video/gst_video_capture.h
        common/utils.h
        common/data_buffer.h
        common/types.cpp
        classifier/classifier.cpp
        classifier/classifier.h)

if (ENABLE_FP16)
  set(CAFFE_SOURCE_FILES classifier/caffe_fp16_classifier.h classifier/caffe_fp16_classifier.cpp)
else ()
  set(CAFFE_SOURCE_FILES classifier/caffe_v1_classifier.cpp classifier/caffe_v1_classifier.h)
endif ()

set(TX1DNN_CAFFE_SOURCE_FILES ${CAFFE_SOURCE_FILES} ${COMMON_SOURCE_FILES})

add_library(tx1dnn-caffe ${TX1DNN_CAFFE_SOURCE_FILES})

target_link_libraries(tx1dnn-caffe
        ${Caffe_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
        ${COMMON_LIBRARIES})

if (Cnmem_FOUND)
  target_link_libraries(tx1dnn-caffe ${Cnmem_LIBRARIES})
endif ()

set(TX1DNN_GIE_SOURCE_FILES
        classifier/gie_classifier.cpp
        classifier/gie_classifier.h
        classifier/gie_inferer.cpp
        classifier/gie_inferer.h
        ${COMMON_SOURCE_FILES})
add_library(tx1dnn-gie ${TX1DNN_GIE_SOURCE_FILES})
target_link_libraries(tx1dnn-gie
        ${GIE_LIBRARIES}
        ${COMMON_LIBRARIES})

set(MXNET_SOURCE_FILES classifier/mxnet_classifier.cpp classifier/mxnet_classifier.h)
add_library(tx1dnn-mxnet ${MXNET_SOURCE_FILES})
target_link_libraries(tx1dnn-mxnet
        ${MXNet_LIBRARIES}
        ${COMMON_LIBRARIES}
        )

target_include_directories(tx1dnn-caffe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tx1dnn-gie PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tx1dnn-mxnet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set Caffe compile flags
function(add_caffe_cflags TARGET)
  SET(CAFFE_CFLAGS "-MMD -MP -pthread -fPIC -O2 -DUSE_OPENCV")

  if (ON_TEGRA)
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

add_caffe_cflags(tx1dnn-caffe)