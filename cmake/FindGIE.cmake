# Find GIE headers and libraries
unset(GIE_FOUND)

set(include_hints_path ${GIE_HOME}/include)
set(lib_hints_path ${GIE_HOME}/lib)

find_path(GIE_INCLUDE_DIRS NAMES Infer.h caffeParser.h
    HINTS
    ${include_hints_path})

find_library(GIE_NVINFER_LIB
    NAMES nvinfer
    HINTS
    ${lib_hints_path})

find_library(GIE_NVCAFFE_PARSER_LIB
    NAMES nvcaffe_parser
    HINTS
    ${lib_hints_path})

find_library(GIE_WCONV_LIB
    NAMES wconv
    HINTS
    ${lib_hints_path})

find_library(GIE_PROTOBUF_LIB
    NAMES protobuf
    HINTS
    ${lib_hints_path})

set(GIE_LIBRARIES ${GIE_NVINFER_LIB} ${GIE_NVCAFFE_PARSER_LIB} ${GIE_WCONV_LIB} ${GIE_PROTOBUF_LIB})
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GIE DEFAULT_MSG
    GIE_INCLUDE_DIRS GIE_LIBRARIES )

if (GIE_INCLUDE_DIRS AND GIE_LIBRARIES)
  set(GIE_FOUND 1)
else()
  message("GIE not found")
endif()