include_directories(${Caffe_INCLUDE_DIRS}
        /usr/local/include)

set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp GstVideoCapture.h CaffeClassifier.cpp CaffeClassifier.h Timer.h CaffeFp16Classifier.cpp CaffeFp16Classifier.h)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
		${Caffe_LIBRARIES}
		${Cnmem_LIBRARIES}
		)

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set Caffe compile flags
function(add_caffe_cflags TARGET)
	SET(CAFFE_CFLAGS "-MMD -MP -pthread -fPIC -DNDEBUG -O2 -DUSE_CUDNN -DUSE_CNMEM -DUSE_OPENCV")
	get_target_property(TEMP ${TARGET} COMPILE_FLAGS)
	if(TEMP STREQUAL "TEMP-NOTFOUND")
		SET(TEMP "") # set to empty string
	endif()
	# append our values
	SET(TEMP "${CAFFE_CFLAGS} ${TEMP}" )
	set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TEMP} )
endfunction()

add_caffe_cflags(tx1dnn)