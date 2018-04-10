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

MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  target_link_libraries(${app} litesql)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/litesql/include)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

# Generate stubs for litesql database library
set(framesdb_sources
  "${CMAKE_CURRENT_BINARY_DIR}/framesdb.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/framesdb.hpp"
  )

add_custom_command(
  OUTPUT ${framesdb_sources}
  COMMAND litesql-gen -t c++
    --output-dir=${CMAKE_CURRENT_BINARY_DIR} framesdb.xml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_source_files_properties(${framesdb_sources}
  PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)

add_executable(train_detector EXCLUDE_FROM_ALL
  # Add back the train_detector.cpp file once it exists.
  # train_detector.cpp
  litesql_writer.cpp
  litesql_writer.h
  ${framesdb_sources})

# Re-enable building the train_detector executable once the train_detector.cpp
# file exists.
# BUILD_ONE(train_detector)
