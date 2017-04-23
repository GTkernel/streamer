cmake_minimum_required(VERSION 2.6)
project(struck)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    src
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# add all .cpp and .h files in src
file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp)
file(GLOB_RECURSE MAIN_CPP ${PROJECT_SOURCE_DIR}/src/main.cpp)
list(REMOVE_ITEM SRC ${MAIN_CPP})
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} src/*.h)

add_library(struck SHARED ${HEADERS} ${SRC})
add_executable(struck.bin ${MAIN_CPP})

target_link_libraries(struck.bin
    struck
    ${OpenCV_LIBS}
)
