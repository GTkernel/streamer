cmake_minimum_required(VERSION 3.2)
project(tx1dnn)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(src)
add_subdirectory(3rdparty)

find_package(GStreamer REQUIRED gstreamer-app)
find_package(GLIB COMPONENTS gobject)
find_package(OpenCV)
find_package(Glog)

include_directories(
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GTK2_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS}
        ${GSTREAMER_APP_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS})

set(MAIN_SOURCES main.cpp)

add_executable(tx1_run_alex ${MAIN_SOURCES})
target_link_libraries(tx1_run_alex
        ${GSTREAMER_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${GSTREAMER_APP_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        tx1dnn)