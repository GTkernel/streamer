cmake_minimum_required(VERSION 2.8)
project(tx1_dnn)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(GStreamer REQUIRED gstreamer-video)
find_package(GLIB COMPONENTS gobject)

include_directories(
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GTK2_INCLUDE_DIRS}
        ${GSTREAMER_VIDEO_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)

add_executable(tx1_dnn ${SOURCE_FILES})
target_link_libraries(tx1_dnn
        ${GSTREAMER_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${GSTREAMER_VIDEO_LIBRARIES})