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

if (NOT USE_TENSORFLOW)
  return ()
endif ()

MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

add_executable(split_nne EXCLUDE_FROM_ALL split_nne.cpp)
BUILD_ONE(split_nne)

add_executable(split_classifier EXCLUDE_FROM_ALL split_classifier.cpp)
BUILD_ONE(split_classifier)

add_executable(classifier EXCLUDE_FROM_ALL classifier.cpp)
BUILD_ONE(classifier)

add_executable(nne EXCLUDE_FROM_ALL nne.cpp)
BUILD_ONE(nne)

add_executable(camera EXCLUDE_FROM_ALL camera.cpp)
BUILD_ONE(camera)
