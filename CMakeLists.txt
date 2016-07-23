cmake_minimum_required(VERSION 3.2)
project(tx1dnn)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(src)
add_subdirectory(3rdparty)

find_package(OpenCV)
find_package(Glog)

include_directories(
    src
    ${GLOG_INCLUDE_DIRS})

set(MAIN_SOURCES main.cpp)

add_executable(tx1_run_alex ${MAIN_SOURCES})
target_link_libraries(tx1_run_alex
    ${OpenCV_LIBRARIES}
    ${GLOG_LIBRARIES}
    tx1dnn)