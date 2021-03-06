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

add_executable(transformer_rpc EXCLUDE_FROM_ALL transformer.cpp)
BUILD_ONE(transformer_rpc)

add_executable(nne_rpc EXCLUDE_FROM_ALL nne.cpp)
BUILD_ONE(nne_rpc)

add_executable(camera_rpc EXCLUDE_FROM_ALL camera.cpp)
BUILD_ONE(camera_rpc)

add_executable(cloud_rpc EXCLUDE_FROM_ALL cloud.cpp)
BUILD_ONE(cloud_rpc)

add_executable(nne1_nocam EXCLUDE_FROM_ALL nne1_nocam.cpp)
BUILD_ONE(nne1_nocam)

add_executable(passer EXCLUDE_FROM_ALL passer.cpp)
BUILD_ONE(passer)

add_executable(nne1 EXCLUDE_FROM_ALL nne1.cpp)
BUILD_ONE(nne1)

add_executable(nne2 EXCLUDE_FROM_ALL nne2.cpp)
BUILD_ONE(nne2)
