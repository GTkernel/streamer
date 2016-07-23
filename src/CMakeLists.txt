set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp GstVideoCapture.h NeuralNet.cpp NeuralNet.h)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${Caffe_LIBRARIES})

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})