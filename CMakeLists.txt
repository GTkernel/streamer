cmake_minimum_required(VERSION 3.2)
project(tx1dnn)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(src)
add_subdirectory(3rdparty)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(GStreamer REQUIRED COMPONENTS gstreamer-app)
find_package(GLIB)

include_directories(
    src
    ${GLOG_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS})

set(MAIN_SOURCES tx1_run_alexnet.cpp)

add_executable(tx1_run_alexnet ${MAIN_SOURCES})
target_link_libraries(tx1_run_alexnet
    ${GLIB_LIBRARIES}
    ${GLIB_GOBJECT_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    tx1dnn)