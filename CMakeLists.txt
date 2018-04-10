# Copyright 2016 The Streamer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0)
project(streamer)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
pkg_check_modules(ZMQ REQUIRED libzmq)
find_package(Boost REQUIRED COMPONENTS
  system filesystem serialization iostreams thread date_time graph)
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

option(USE_CAFFE "Build with Caffe." OFF)
if (USE_CAFFE)
  find_package(Caffe REQUIRED)
  include_directories(SYSTEM ${Caffe_INCLUDE_DIRS})
  add_definitions("-D USE_CAFFE")
endif ()

option(USE_TENSORFLOW "Build with TensorFlow." OFF)
if (USE_TENSORFLOW)
  find_package(TensorFlow REQUIRED)
  find_package(Protobuf REQUIRED)
  include_directories(SYSTEM
    ${TensorFlow_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/3rdparty/nsync/public/
  )
  add_definitions("-D USE_TENSORFLOW")
endif()

option(USE_FRCNN "Build with Caffe Faster Rcnn" OFF)
if (USE_FRCNN)
  add_definitions("-D USE_FRCNN")
  message("Build with Caffe Faster Rcnn")
endif ()

option(USE_DLIB "Build with Dlib" OFF)
if (USE_DLIB)
  add_definitions("-D USE_DLIB")
  message("Build with Dlib")
endif ()

option(USE_NCS "Build with NCS" OFF)
if (USE_NCS)
  find_package(NCS REQUIRED)
  include_directories(${NCS_INCLUDE_DIRS})
  add_definitions("-D USE_NCS")
  message("Build with NCS")
endif ()

option(USE_ATHENA "Build with Athena" OFF)
if (USE_ATHENA)
  find_package(Athena REQUIRED)
  include_directories(${Athena_INCLUDE_DIRS})
  include_directories(${Athena_utils_INCLUDE_DIRS})
  add_definitions("-D USE_ATHENA")
  message("Build with Athena")
endif ()

option(USE_SSD "Build with Caffe SSD" OFF)
if (USE_SSD)
  add_definitions("-D USE_SSD")
  message("Build with Caffe SSD")
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

# Common include paths
include_directories(src)
include_directories(SYSTEM
  3rdparty
  ${GLOG_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ZMQ_INCLUDE_DIRS}
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
