# Find GIE headers and libraries
unset(GIE_FOUND)

find_path(GIE_INCLUDE_DIRS NAMES Infer.h caffeParser.h
    HINTS
    ${GIE_HOME}/include
    /usr/local/include
    /usr/include)

find_library(GIE_LIBRARIES NAMES nvinfer nvcaffe_parser protobuf wconv
    HINTS
    ${GIE_HOME}/lib
    /usr/local/lib
    /usr/lib)

if (GIE_INCLUDE_DIRS AND GIE_LIBRARIES)
  set(GIE_FOUND 1)
else()
  message("GIE not found")
endif()