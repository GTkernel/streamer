find_package(GStreamer REQUIRED CMPONENTS gstreamer-app)
find_package(GLIB COMPONENTS gobject)

include_directories(
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS})

set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp GstVideoCapture.h)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn)

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})