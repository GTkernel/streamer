include_directories(${Caffe_INCLUDE_DIRS}
        /usr/local/include)

set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp GstVideoCapture.h CaffeClassifier.cpp CaffeClassifier.h)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${Boost_SYSTEM_LIBRARY_RELEASE}
		${CUDA_LIBRARIES}
        ${Caffe_LIBRARIES}
		# Hard code these for now, dynamically detect in the future
		/home/ubuntu/Code/caffe/3rdparty/cnmem/build/libcnmem.so
		/home/ubuntu/Code/caffe/3rdparty/cnmem/build/libcnmem.so.1
		/home/ubuntu/Code/caffe/3rdparty/cnmem/build/libcnmem.so.1.0.0)

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
