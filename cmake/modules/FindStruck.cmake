unset(Struck_FOUND)

find_path(Struck_INCLUDE_DIRS NAMES src/Tracker.h
    HINTS
	${STRUCK_HOME}
    /usr/local/include)

find_library(Struck_LIBRARIES NAMES struck
    HINTS
	${STRUCK_HOME}/build
    /usr/local/lib)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Struck DEFAULT_MSG
    Struck_INCLUDE_DIRS Struck_LIBRARIES)

if (Struck_LIBRARIES AND Struck_INCLUDE_DIRS)
    set(Struck_FOUND 1)
else ()
    message("Not found Struck")
endif ()
