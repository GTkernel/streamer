# Athena package for DbWriter
unset(Athena_FOUND)

find_path(Athena_INCLUDE_DIRS NAMES client/AthenaClient.h
  HINTS
  ${Athena_HOME})

find_path(Athena_utils_INCLUDE_DIRS NAMES comm/Connection.h
  HINTS
  ${Athena_HOME}/utils/include)

find_library(Athena_LIBRARIES NAMES athena-client
  HINTS
  ${Athena_HOME}/client)

find_library(Athena_utils_LIBRARIES NAMES athena-utils
  HINTS
  ${Athena_HOME}/utils)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Athena DEFAULT_MSG
  Athena_INCLUDE_DIRS Athena_utils_INCLUDE_DIRS Athena_LIBRARIES Athena_utils_LIBRARIES)

if (Athena_LIBRARIES AND Athena_INCLUDE_DIRS)
  set(Athena_FOUND 1)
else ()
  message("Not found Athena")
endif ()
