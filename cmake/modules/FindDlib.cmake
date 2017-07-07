# Dlib package for tracker
unset(Dlib_FOUND)

find_path(Dlib_INCLUDE_DIRS NAMES dlib/any.h
  HINTS
  ${Dlib_HOME})

find_library(Dlib_LIBRARIES NAMES dlib
  HINTS
  ${Dlib_HOME}/dlib/build)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Dlib DEFAULT_MSG
  Dlib_INCLUDE_DIRS Dlib_LIBRARIES)

if (Dlib_LIBRARIES AND Dlib_INCLUDE_DIRS)
  set(Dlib_FOUND 1)
else ()
  message("Not found Dlib")
endif ()
