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

MACRO(ADD_BUILD_REQS target)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -fopenmp=libomp)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -Wall -Wextra -fopenmp)
  endif()
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
ENDMACRO ()

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  message("Using compiler cache: ${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

# Increase the number of types allowed in a boost::variant. See
# https://stackoverflow.com/questions/34702612/how-to-increase-the-number-of-types-that-can-handled-by-boostvariant
add_definitions("-D BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS")
add_definitions("-D BOOST_MPL_LIMIT_LIST_SIZE=30")
