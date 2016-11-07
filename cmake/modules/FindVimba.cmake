# Find Vimba SDK. These variables will be set:
#
# Vimba_FOUND        - If Vimba FlyCapture SDK is found
# Vimba_INCLUDE_DIRS - Header directories for Vimba FlyCapTure SDK
# Vimba_LIBRARIES    - Libraries for Vimba FlyCapture SDK

unset(Vimba_FOUND)

find_path(Vimba_INCLUDE_DIRS NAMES VimbaCPP/Include/VimbaCPP.h
  HINTS
  ${Vimba_HOME}/include)

#find_library(Vimba_C_LIBRARY NAMES VimbaC
#  HINTS
#  ${Vimba_HOME}/lib)
#
#find_library(Vimba_CPP_LIBRARY NAMES VimbaCPP
#  HINTS
#  ${Vimba_HOME}/lib)

find_library(Vimba_LIBRARIES NAMES VimbaCPP VimbaC
  HINTS
  ${Vimba_HOME}/lib)

set(Vimba_LIBRARIES ${Vimba_CPP_LIBRARY} ${Vimba_C_LIBRARY})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Vimba DEFAULT_MSG
  Vimba_INCLUDE_DIRS Vimba_LIBRARIES)

message(${Vimba_LIBRARIES})

if (Vimba_INCLUDE_DIRS AND Vimba_LIBRARIES)
  set(Vimba_FOUND 1)
else ()
  message("Not found Vimba")
endif ()
