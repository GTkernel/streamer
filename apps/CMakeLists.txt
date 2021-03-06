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

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

if (USE_RPC)
  include_directories(SYSTEM ${CMAKE_BINARY_DIR}/src)
endif ()

# Macro to get sub directories
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH (child ${children})
    IF (IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF ()
  ENDFOREACH ()
  SET(${result} ${dirlist})
ENDMACRO()

MACRO(BUILD_APP app_dir)
  FILE(GLOB app_srcs ${app_dir}/*.cpp ${app_dir}/*.h ${app_dir}/*.c)
  get_filename_component(app_name ${app_dir} NAME_WE)
  add_executable(${app_name} EXCLUDE_FROM_ALL ${app_srcs})
  if (JEMALLOC_FOUND)
    target_link_libraries(${app_name} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app_name} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  ADD_BUILD_REQS(${app_name})
ENDMACRO()

set(APPS_ROOT ${PROJECT_SOURCE_DIR}/apps/)
SUBDIRLIST(APP_DIRS ${APPS_ROOT})

add_custom_target(apps)
foreach (app_dir ${APP_DIRS})
  if (EXISTS ${PROJECT_SOURCE_DIR}/apps/${app_dir}/CMakeLists.txt)
    add_subdirectory(${app_dir})
  else ()
    BUILD_APP(${app_dir})
    get_filename_component(app_name ${app_dir} NAME_WE)
    add_dependencies(apps ${app_name})
  endif ()
endforeach ()
