cmake_minimum_required(VERSION 3.2)
project(tx1dnn)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(GStreamer REQUIRED COMPONENTS gstreamer-app)
find_package(GLIB REQUIRED gobject)
find_package(Caffe REQUIRED)
find_package(CBLAS)
find_package(Boost REQUIRED COMPONENTS system)
find_package(CUDA)

if(APPLE)
    find_package(vecLib)
endif()

include_directories(
    src
    ${GLOG_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS})

if(APPLE)
    include_directories(${vecLib_INCLUDE_DIR})
endif()

set(MAIN_SOURCES apps/tx1_run_alexnet.cpp)

add_subdirectory(src)
add_subdirectory(3rdparty)
add_subdirectory(apps)
