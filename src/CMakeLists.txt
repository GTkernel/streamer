find_package(GStreamer REQUIRED gstreamer-app)
find_package(GLIB COMPONENTS gobject)

include_directories(
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GTK2_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS})

set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn
        ${GSTREAMER_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES})

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})