################################################################################################
# tx1dnn status report function. (Borrowed from Caffe)
# Automatically align right column and selects text based on condition.
# Usage:
#   tx1dnn_status(<text>)
#   tx1dnn_status(<heading> <value1> [<value2> ...])
#   tx1dnn_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(tx1dnn_status text)
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
# Function for fetching tx1dnn version from git and headers
# Usage:
#   tx1dnn_extract_tx1dnn_version()
function(tx1dnn_extract_tx1dnn_version)
  set(tx1dnn_GIT_VERSION "unknown")
  find_package(Git)
  if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE tx1dnn_GIT_VERSION
      RESULT_VARIABLE __git_result)
    if (NOT ${__git_result} EQUAL 0)
      set(tx1dnn_GIT_VERSION "unknown")
    endif ()
  endif ()

  set(tx1dnn_GIT_VERSION ${tx1dnn_GIT_VERSION} PARENT_SCOPE)
  set(tx1dnn_VERSION "<TODO> (tx1dnn doesn't declare its version in headers)" PARENT_SCOPE)

  # tx1dnn_parse_header(${tx1dnn_INCLUDE_DIR}/tx1dnn/version.hpp tx1dnn_VERSION_LINES tx1dnn_MAJOR tx1dnn_MINOR tx1dnn_PATCH)
  # set(tx1dnn_VERSION "${tx1dnn_MAJOR}.${tx1dnn_MINOR}.${tx1dnn_PATCH}" PARENT_SCOPE)

  # or for #define tx1dnn_VERSION "x.x.x"
  # tx1dnn_parse_header_single_define(tx1dnn ${tx1dnn_INCLUDE_DIR}/tx1dnn/version.hpp tx1dnn_VERSION)
  # set(tx1dnn_VERSION ${tx1dnn_VERSION_STRING} PARENT_SCOPE)

endfunction()

################################################################################################
# Function merging lists of compiler flags to single string.
# Usage:
#   tx1dnn_merge_flag_lists(out_variable <list1> [<list2>] [<list3>] ...)
function(tx1dnn_merge_flag_lists out_var)
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
# Prints accumulated tx1dnn configuration summary
# Usage:
#   tx1dnn_print_configuration_summary()

function(tx1dnn_print_configuration_summary)
  tx1dnn_extract_tx1dnn_version()
  set(tx1dnn_VERSION ${tx1dnn_VERSION} PARENT_SCOPE)

  tx1dnn_merge_flag_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS)
  tx1dnn_merge_flag_lists(__flags_deb CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS)

  tx1dnn_status("")
  tx1dnn_status("******************* tx1dnn Configuration Summary *******************")
  tx1dnn_status("General:")
  tx1dnn_status("  Version           :   ${tx1dnn_TARGET_VERSION}")
  tx1dnn_status("  Git               :   ${tx1dnn_GIT_VERSION}")
  tx1dnn_status("  System            :   ${CMAKE_SYSTEM_NAME}")
  tx1dnn_status("  On Tegra          : " TEGRA THEN "Yes" ELSE "No")
  tx1dnn_status("  Compiler          :   ${CMAKE_CXX_COMPILER} (${COMPILER_FAMILY} ${COMPILER_VERSION})")
  tx1dnn_status("  Release CXX flags :   ${__flags_rel}")
  tx1dnn_status("  Debug CXX flags   :   ${__flags_deb}")
  tx1dnn_status("  Build type        :   ${CMAKE_BUILD_TYPE}")
  tx1dnn_status("")
  tx1dnn_status("Options:")
  tx1dnn_status("  USE_CAFE          :   ${USE_CAFFE}")
  tx1dnn_status("  USE_MXNE          :   ${USE_MXNET}")
  tx1dnn_status("  USE_GIE           :   ${USE_GIE}")
  tx1dnn_status("  USE_FP16          :   ${USE_FP16}")
  tx1dnn_status("  BACKEND           :   ${BACKEND}")
  tx1dnn_status("  USE_PTGRAY        :   ${USE_PTGRAY}")
  tx1dnn_status("")
  tx1dnn_status("Frameworks:")
  tx1dnn_status("  Caffe             : " Caffe_FOUND THEN "Yes (${CAFFE_HOME})" ELSE "No")
  tx1dnn_status("  MXNet             : " MXNet_FOUND THEN "Yes (${MXNET_HOME})" ELSE "No")
  tx1dnn_status("  GIE               : " GIE_FOUND THEN "Yes (${GIE_HOME})" ELSE "No")
  tx1dnn_status("")
  tx1dnn_status("Dependencies:")
  tx1dnn_status("  Linker flags      :   ${CMAKE_EXE_LINKER_FLAGS}")
  tx1dnn_status("  Boost             :   Yes (ver. ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION})")
  tx1dnn_status("  OpenCV            :   Yes (ver. ${OpenCV_VERSION})")
  tx1dnn_status("  GStreamer         :   Yes")
  tx1dnn_status("  GStreamer-app     :   Yes")
  tx1dnn_status("  glog              :   Yes")
  tx1dnn_status("  gflags            :   Yes")
  tx1dnn_status("  VecLib            : " VECLIB_FOUND THEN "Yes" ELSE "No")
  tx1dnn_status("  CUDA              : " CUDA_FOUND THEN "Yes (ver. ${CUDA_VERSION_STRING})" ELSE "No")
  tx1dnn_status("  Cnmem             : " Cnmem_FOUND THEN "Yes" ELSE "No")
  tx1dnn_status("  OpenCL            : " OPENCL_FOUND THEN "Yes (ver. ${OPENCL_VERSION_STRING})" ELSE "No")
  tx1dnn_status("  HDF5              : " HDF5_FOUND THEN "Yes" ELSE "No")
  tx1dnn_status("  Protobuf          : " PROTOBUF_FOUND THEN " Yes" ELSE "No")
  tx1dnn_status("  JeMalloc          : " JEMALLOC_FOUND THEN " Yes" ELSE "No")
  tx1dnn_status("  FlyCapture        : " PtGray_FOUND THEN " Yes (${PtGray_INCLUDE_DIRS})" ELSE "No")

  tx1dnn_status("")
  tx1dnn_status("Install:")
  tx1dnn_status(" Install path : ${CMAKE_INSTALL_PREFIX}")
  tx1dnn_status(" ")
endfunction()