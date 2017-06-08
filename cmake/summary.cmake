################################################################################################
# streamer status report function. (Borrowed from Caffe)
# Automatically align right column and selects text based on condition.
# Usage:
#   streamer_status(<text>)
#   streamer_status(<heading> <value1> [<value2> ...])
#   streamer_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(streamer_status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach (arg ${ARGN})
    if (arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif (arg STREQUAL "ELSE")
      set(status_current_name "else")
    else ()
      list(APPEND status_${status_current_name} ${arg})
    endif ()
  endforeach ()

  if (DEFINED status_cond)
    set(status_placeholder_length 23)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if (status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif (DEFINED status_then OR DEFINED status_else)
      message(STATUS "${text}")
      set(status_text "${status_placeholder}")
    else ()
      set(status_text "${text}")
    endif ()

    if (DEFINED status_then OR DEFINED status_else)
      if (${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        message(STATUS "${status_text} ${status_then}")
      else ()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        message(STATUS "${status_text} ${status_else}")
      endif ()
    else ()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      message(STATUS "${status_text} ${status_cond}")
    endif ()
  else ()
    message(STATUS "${text}")
  endif ()
endfunction()


################################################################################################
# Function for fetching streamer version from git and headers
# Usage:
#   streamer_extract_streamer_version()
function(streamer_extract_streamer_version)
  set(streamer_GIT_VERSION "unknown")
  find_package(Git)
  if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE streamer_GIT_VERSION
      RESULT_VARIABLE __git_result)
    if (NOT ${__git_result} EQUAL 0)
      set(streamer_GIT_VERSION "unknown")
    endif ()
  endif ()

  set(streamer_GIT_VERSION ${streamer_GIT_VERSION} PARENT_SCOPE)
  set(streamer_VERSION "<TODO> (streamer doesn't declare its version in headers)" PARENT_SCOPE)

  # streamer_parse_header(${streamer_INCLUDE_DIR}/streamer/version.hpp streamer_VERSION_LINES streamer_MAJOR streamer_MINOR streamer_PATCH)
  # set(streamer_VERSION "${streamer_MAJOR}.${streamer_MINOR}.${streamer_PATCH}" PARENT_SCOPE)

  # or for #define streamer_VERSION "x.x.x"
  # streamer_parse_header_single_define(streamer ${streamer_INCLUDE_DIR}/streamer/version.hpp streamer_VERSION)
  # set(streamer_VERSION ${streamer_VERSION_STRING} PARENT_SCOPE)

endfunction()

################################################################################################
# Function merging lists of compiler flags to single string.
# Usage:
#   streamer_merge_flag_lists(out_variable <list1> [<list2>] [<list3>] ...)
function(streamer_merge_flag_lists out_var)
  set(__result "")
  foreach (__list ${ARGN})
    foreach (__flag ${${__list}})
      string(STRIP ${__flag} __flag)
      set(__result "${__result} ${__flag}")
    endforeach ()
  endforeach ()
  string(STRIP ${__result} __result)
  set(${out_var} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Prints accumulated streamer configuration summary
# Usage:
#   streamer_print_configuration_summary()

function(streamer_print_configuration_summary)
  streamer_extract_streamer_version()
  set(streamer_VERSION ${streamer_VERSION} PARENT_SCOPE)

  streamer_merge_flag_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS)
  streamer_merge_flag_lists(__flags_deb CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS)

  streamer_status("")
  streamer_status("******************* streamer Configuration Summary *******************")
  streamer_status("General:")
  streamer_status("  Version           :   ${streamer_TARGET_VERSION}")
  streamer_status("  Git               :   ${streamer_GIT_VERSION}")
  streamer_status("  System            :   ${CMAKE_SYSTEM_NAME}")
  streamer_status("  On Tegra          : " TEGRA THEN "Yes" ELSE "No")
  streamer_status("  Compiler          :   ${CMAKE_CXX_COMPILER} (${COMPILER_FAMILY} ${COMPILER_VERSION})")
  streamer_status("  Release CXX flags :   ${__flags_rel}")
  streamer_status("  Debug CXX flags   :   ${__flags_deb}")
  streamer_status("  Build type        :   ${CMAKE_BUILD_TYPE}")
  streamer_status("")
  streamer_status("Options:")
  streamer_status("  BACKEND           :   ${BACKEND}")
  streamer_status("")
  streamer_status("  USE_CAFFE         :   ${USE_CAFFE}")
  streamer_status("  USE_MXNET         :   ${USE_MXNET}")
  streamer_status("  USE_GIE           :   ${USE_GIE}")
  streamer_status("  USE_FP16          :   ${USE_FP16}")
  streamer_status("  USE_PTGRAY        :   ${USE_PTGRAY}")
  streamer_status("  USE_VIMBA         :   ${USE_VIMBA}")
  streamer_status("  USE_RPC           :   ${USE_RPC}")
  streamer_status("  USE_ZMQ           :   ${USE_ZMQ}")
  streamer_status("")
  streamer_status("Frameworks:")
  streamer_status("  Caffe             : " Caffe_FOUND THEN "Yes (${CAFFE_HOME})" ELSE "No")
  streamer_status("  MXNet             : " MXNet_FOUND THEN "Yes (${MXNET_HOME})" ELSE "No")
  streamer_status("  GIE               : " GIE_FOUND THEN "Yes (${GIE_HOME})" ELSE "No")
  streamer_status("")
  streamer_status("Dependencies:")
  streamer_status("  Linker flags      :   ${CMAKE_EXE_LINKER_FLAGS}")
  streamer_status("  Boost             :   Yes (ver. ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION})")
  streamer_status("  OpenCV            :   Yes (ver. ${OpenCV_VERSION})")
  streamer_status("  Eigen             :   Yes (ver. ${EIGEN3_VERSION})")
  streamer_status("  GStreamer         :   Yes")
  streamer_status("  GStreamer-app     :   Yes")
  streamer_status("  glog              :   Yes")
  streamer_status("  gflags            :   Yes")
  streamer_status("  VecLib            : " VECLIB_FOUND THEN "Yes" ELSE "No")
  streamer_status("  CUDA              : " CUDA_FOUND THEN "Yes (ver. ${CUDA_VERSION_STRING})" ELSE "No")
  streamer_status("  Cnmem             : " Cnmem_FOUND THEN "Yes" ELSE "No")
  streamer_status("  OpenCL            : " OPENCL_FOUND THEN "Yes (ver. ${OPENCL_VERSION_STRING})" ELSE "No")
  streamer_status("  HDF5              : " HDF5_FOUND THEN "Yes" ELSE "No")
  if (USE_CAFFE OR USE_RPC)
    streamer_status("  Protobuf          : " PROTOBUF_FOUND THEN " Yes" ELSE "No")
  endif ()
  if (USE_RPC)
    streamer_status("  gRPC              : " GRPC_LIBRARIES THEN " Yes (${GRPC_LIBRARIES})" ELSE "No")
  endif ()
  if (USE_ZMQ)
    streamer_status("  ZeroMQ            : " ZMQ_FOUND THEN " Yes (${ZMQ_LIBRARIES})" ELSE "No")
  endif ()
  streamer_status("  JeMalloc          : " JEMALLOC_FOUND THEN " Yes" ELSE "No")
  streamer_status("  FlyCapture        : " PtGray_FOUND THEN " Yes (${PtGray_INCLUDE_DIRS})" ELSE "No")
  streamer_status("  Vimba             : " Vimba_FOUND THEN " Yes (${Vimba_INCLUDE_DIRS})" ELSE "No")

  streamer_status("")
  streamer_status("Install:")
  streamer_status(" Install path : ${CMAKE_INSTALL_PREFIX}")
  streamer_status(" ")
endfunction()
