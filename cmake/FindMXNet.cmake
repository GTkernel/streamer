# MXNet package
unset(NXNet_FOUND)

find_path(MXNet_INCLUDE_DIRS NAMES mxnet/io.h mxnet/kvstore.h mxnet/ndarray.h
        HINTS
        ${MXNET_HOME}/include
        /usr/local/include)

find_library(MXNet_LIBRARIES NAMES mxnet
        HINTS
        ${MXNET_HOME}/lib
        /usr/local/lib)

find_package(OpenCV REQUIRED)
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MXNet DEFAULT_MSG
        MXNet_INCLUDE_DIRS MXNet_LIBRARIES)

if (MXNet_LIBRARIES AND MXNet_INCLUDE_DIRS)
  set(MXNet_FOUND 1)
else ()
  message("Not found MXNet")
endif ()