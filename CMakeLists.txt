cmake_minimum_required(VERSION 3.0)
project(streamer)

include(cmake/summary.cmake)
include(cmake/clangformat.cmake)
include(cmake/build.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

# See https://github.com/opencv/opencv/issues/6542
set(CUDA_USE_STATIC_CUDA_RUNTIME false)

if (APPLE)
  set(CMAKE_MACOSX_RPATH true)
endif ()

# Prevent cmake from caching Caffe paths
unset(Caffe_FOUND CACHE)
unset(Caffe_INCLUDE_DIRS CACHE)
unset(Caffe_LIBRARIES CACHE)

# Find libraries
find_package(PkgConfig REQUIRED)
find_package(OpenCV 3.0 REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)
pkg_check_modules(GLOG REQUIRED libglog)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(Boost REQUIRED COMPONENTS
  system filesystem serialization iostreams thread date_time)
find_package(JeMalloc)

# Determine if is on Tegra
if (EXISTS /var/libopencv4tegra-repo)
  SET(TEGRA true)
  add_definitions("-D TEGRA")
endif ()

if (APPLE)
  find_package(vecLib)
  add_definitions("-D APPLE")
endif ()

# Different backends
set(USE_CUDA false)
set(USE_OPENCL false)
set(USE_CPU false)

option(BACKEND "device backend to run the program: cpu, cuda or opencl" cpu)
if (BACKEND STREQUAL "cuda")
  set(USE_CUDA true)
  find_package(CUDA REQUIRED)
  find_package(Cnmem)
  add_definitions("-D USE_CUDA")
elseif (BACKEND STREQUAL "opencl")
  find_package(OpenCL REQUIRED)
  set(USE_OPENCL true)
  add_definitions("-D USE_OPENCL")
elseif (BACKEND STREQUAL "cpu")
  set(USE_CPU true)
  add_definitions("-D CPU_ONLY")
else ()
  message(FATAL_ERROR "Backend ${BACKEND} is invalid")
endif ()

option(USE_CAFFE "Build with Caffe" ON)
if (USE_CAFFE)
  find_package(Caffe REQUIRED)
  include_directories(SYSTEM ${Caffe_INCLUDE_DIRS})
  add_definitions("-D USE_CAFFE")
endif ()

option(USE_PTGRAY "Build with PtGray GigE SDK or not." OFF)
if (USE_PTGRAY)
  find_package(PtGray REQUIRED)
  include_directories(SYSTEM ${PtGray_FC_INCLUDE_DIRS})
  add_definitions("-D USE_PTGRAY")
endif()

option(USE_VIMBA "Build with AlliedVision Vimba SDK or not." OFF)
if (USE_VIMBA)
  find_package(Vimba REQUIRED)
  include_directories(SYSTEM ${Vimba_INCLUDE_DIRS})
  add_definitions("-D USE_VIMBA")
endif()

option(USE_RPC "Build RPC support." OFF)
if (USE_RPC)
  find_package(Protobuf REQUIRED)
  find_package(GRPC REQUIRED)
  include_directories(
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
  )
  add_definitions("-D USE_RPC")
endif()

option(USE_ZMQ "Build ZeroMQ support." OFF)
if (USE_ZMQ)
  pkg_check_modules(ZMQ REQUIRED libzmq)
  add_definitions("-D USE_ZMQ")
endif ()

# Common include paths
include_directories(src)
include_directories(SYSTEM
  3rdparty
  ${GLOG_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  )

# Include cblas libraries
if (CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif ()

if (APPLE)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
endif ()

add_subdirectory(src)
add_subdirectory(3rdparty)
add_subdirectory(apps)

# Enable CTest
option(TEST_ON "Build with test cases" OFF)
if (TEST_ON)
  enable_testing()
  add_subdirectory(test)
endif ()

# Copy config.example files, if corresponding config file is not there, cp toml.example to .toml
file(GLOB EXAMPLE_CONFIG_FILES ${PROJECT_SOURCE_DIR}/config/*.toml.example)
foreach (f ${EXAMPLE_CONFIG_FILES})
  get_filename_component(basename ${f} NAME_WE)
  get_filename_component(dir ${f} DIRECTORY)
  file(COPY ${f} DESTINATION config)
  configure_file(${f} config/${basename}.toml)
endforeach ()

file(GLOB CONFIG_FILES ${PROJECT_SOURCE_DIR}/config/*.toml)
foreach (f ${CONFIG_FILES})
  file(COPY ${f} DESTINATION config)
endforeach ()

streamer_print_configuration_summary()
