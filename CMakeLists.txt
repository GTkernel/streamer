cmake_minimum_required(VERSION 3.2)
project(tx1dnn)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Find libraries
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(GStreamer REQUIRED COMPONENTS gstreamer-app)
find_package(GLIB REQUIRED gobject)
find_package(Caffe REQUIRED)
find_package(GIE REQUIRED)
find_package(Protobuf 3.0 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(MXNet)

# Determine platform
if (EXISTS /var/libopencv4tegra-repo)
    message("Build on Tegra")
    SET(ON_TEGRA 1)
    add_definitions("-DON_TEGRA")
endif()

if (ON_TEGRA)
    find_package(Cnmem)
    find_package(CUDA)
endif()

# Only used for development convienience, this project is not supposed to build or run on platform other than TX1
if (APPLE)
    message("Build on Mac")
    find_package(vecLib)
    SET(ON_MAC 1)
    add_definitions("-D ON_MAC")
    add_definitions("-D CPU_ONLY")
endif()

if (USE_GIE)
    message("Build using GIE")
    add_definitions("-D USE_GIE")
endif()

include_directories(
    src
    ${GLOG_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
    ${GIE_INCLUDE_DIRS}
)

if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (ON_MAC)
    include_directories(${vecLib_INCLUDE_DIR})
endif()

add_subdirectory(src)
add_subdirectory(3rdparty)
add_subdirectory(apps)