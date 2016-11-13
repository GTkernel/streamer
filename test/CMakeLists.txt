include_directories(${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)
include_directories(${gmock_SOURCE_DIR} ${gmock_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/test/include)

file(GLOB_RECURSE TEST_SRCS ${PROJECT_SOURCE_DIR}/test/test*.cpp)

add_library(streamer_gtest_main streamer_gtest_main.cpp)
target_link_libraries(streamer_gtest_main streamer gtest)

add_custom_target(tests)
set(TEST_CONFIG_DIR ${PROJECT_BINARY_DIR}/test/config)
foreach (f ${TEST_SRCS})
  get_filename_component(basename ${f} NAME_WE)
  add_executable(${basename} EXCLUDE_FROM_ALL ${f})
  target_link_libraries(${basename} streamer gtest streamer_gtest_main gmock gmock_main)
  add_test(${basename} ${basename} ${TEST_CONFIG_DIR})
  add_dependencies(tests ${basename})
endforeach ()

file(GLOB TEST_CONFIG_FILES ${PROJECT_SOURCE_DIR}/test/config/*.toml)
foreach (f ${TEST_CONFIG_FILES})
  file(COPY ${f} DESTINATION config)
endforeach ()