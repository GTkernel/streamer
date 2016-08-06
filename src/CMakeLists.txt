set(COMMON_LIBRARIES
	${GLIB_LIBRARIES}
	${GLIB_GOBJECT_LIBRARIES}
	${OpenCV_LIBRARIES}
	${GLOG_LIBRARIES}
	${GSTREAMER_LIBRARIES}
	${GSTREAMER_APP_LIBRARIES}
	${Boost_SYSTEM_LIBRARY})

set(COMMON_SOURCE_FILES gst_video_capture.cpp gst_video_capture.h utils.cpp utils.cpp classifier.cpp classifier.h)

set(TX1DNN_CAFFE_SOURCE_FILES caffe_v1_classifier.cpp caffe_v1_classifier.h ${COMMON_SOURCE_FILES})

add_library(tx1dnn-caffe ${TX1DNN_CAFFE_SOURCE_FILES})

target_link_libraries(tx1dnn-caffe
		${Caffe_LIBRARIES}
		${PROTOBUF_LIBRARIES}
		${COMMON_LIBRARIES})

if (Cnmem_FOUND)
	target_link_libraries(tx1dnn-caffe ${Cnmem_LIBRARIES})
endif()

set(TX1DNN_GIE_SOURCE_FILES gie_classifier.cpp gie_classifier.h gie_inferer.cpp gie_inferer.h float16.cpp float16.h ${COMMON_SOURCE_FILES} data_buffer.h)
add_library(tx1dnn-gie ${TX1DNN_GIE_SOURCE_FILES})
target_link_libraries(tx1dnn-gie
	${GIE_LIBRARIES}
	${COMMON_LIBRARIES})

set(MXNET_SOURCE_FILES mxnet_classifier.cpp mxnet_classifier.h)
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
	endif()

	get_target_property(TEMP ${TARGET} COMPILE_FLAGS)
	if(TEMP STREQUAL "TEMP-NOTFOUND")
		SET(TEMP "") # set to empty string
	endif()
	# append our values
	SET(TEMP "${CAFFE_CFLAGS} ${TEMP}" )
	set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TEMP} )
endfunction()

add_caffe_cflags(tx1dnn-caffe)