cmake_minimum_required(VERSION 3.0)
project(streamer)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# See https://github.com/opencv/opencv/issues/6542
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# If you want to make clion faster, uncomment this following line
# See https://intellij-support.jetbrains.com/hc/en-us/community/posts/207110289-Best-practices-for-CMake-files-for-use-in-CLion
set(CMAKE_CONFIGURATION_TYPES "Debug")


if (APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif ()

# Prevent cmake from caching Caffe paths
unset(Caffe_FOUND CACHE)
unset(Caffe_INCLUDE_DIRS CACHE)
unset(Caffe_LIBRARIES CACHE)
# Find libraries
find_package(OpenCV REQUIRED)
find_package(GStreamer REQUIRED COMPONENTS gstreamer-app)
find_package(Glog REQUIRED)
find_package(GLIB REQUIRED gobject)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(JeMalloc)

# Determine if is on Tegra
if (EXISTS /var/libopencv4tegra-repo)
  message("Build on Tegra")
  SET(TEGRA 1)
  add_definitions("-D TEGRA")
endif ()

if (APPLE)
  message("Build on Mac")
  find_package(vecLib)
  add_definitions("-D APPLE")
endif ()

# Different backends
set(USE_CUDA off)
set(USE_OPENCL off)
SET(USE_CPU off)

option(BACKEND "device backend to run the program: cpu, cuda or opencl" cpu)
if (BACKEND STREQUAL "cuda")
  message("Build with CUDA")
  set(USE_CUDA true)
  find_package(CUDA REQUIRED)
  find_package(Cnmem)
  add_definitions("-D USE_CUDA")
elseif (BACKEND STREQUAL "opencl")
  message("Build with OpenCL")
  find_package(OpenCL REQUIRED)
  set(USE_OPENCL true)
  add_definitions("-D USE_OPENCL")
elseif (BACKEND STREQUAL "cpu")
  message("Build with cpu only.")
  set(USE_CUDA off)
  set(USE_OPENCL off)
  add_definitions("-D CPU_ONLY")
else ()
  message(FATAL_ERROR "Backend ${BACKEND} is invalid")
endif ()

option(USE_FP16 "use FP16 or not, it should only be enabled on Tegra" OFF)
if (USE_FP16)
  if (NOT TEGRA OR NOT USE_CUDA)
    message(FATAL_ERROR "FP16 enabled, but not on Tegra, or not built with CUDA")
  endif ()
  message("Build with FP16")
  add_definitions("-D USE_FP16")
endif ()

option(USE_CAFFE "Build with Caffe" ON)
if (USE_CAFFE)
  find_package(Caffe REQUIRED)
  find_package(HDF5 REQUIRED)
  find_package(Protobuf)
  include_directories(${HDF5_INCLUDE_DIRS})
  include_directories(${Caffe_INCLUDE_DIRS})
  add_definitions("-D USE_CAFFE")
  message("Build with Caffe")
endif ()

option(USE_GIE "Build with GIE or not. GIE could not be used together with Caffe for now." OFF)
if (USE_GIE)
  if (USE_CAFFE)
    message(FATAL_ERROR "GIE can't be built with Caffe")
  endif ()
  find_package(GIE REQUIRED)
  include_directories(${GIE_INCLUDE_DIRS})
  add_definitions("-D USE_GIE")
  message("Build with GIE")
endif ()

option(USE_MXNET "Build with MXNet or not." OFF)
if (USE_MXNET)
  find_package(MXNet REQUIRED)
  include_directories(${MXNet_INCLUDE_DIRS})
  add_definitions("-D USE_MXNET")
  message("Build with MXNet")
endif ()

option(USE_PTGRAY "Build with PtGray GigE SDK or not." OFF)
if (USE_PTGRAY)
  find_package(PtGray REQUIRED)
  include_directories(${PtGray_FC_INCLUDE_DIRS})
  add_definitions("-D USE_PTGRAY")
  message("Build with PtGray")
endif()

# Common include paths
include_directories(
  src
  3rdparty
  ${GLOG_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GSTREAMER_APP_INCLUDE_DIRS}
  ${GSTREAMER_}
)

# Include cblas libraries
if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
endif ()

if (APPLE)
  include_directories(${vecLib_INCLUDE_DIR})
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

include(cmake/summary.cmake)
include(cmake/clangformat.cmake)
streamer_print_configuration_summary()
