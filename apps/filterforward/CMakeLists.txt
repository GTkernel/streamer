if (NOT USE_TENSORFLOW)
  return ()
endif ()

find_package(CURL REQUIRED)

MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

add_executable(filterforward EXCLUDE_FROM_ALL filterforward.cpp)
include_directories(${CURL_INCLUDE_DIR})
target_link_libraries(filterforward ${CURL_LIBRARIES})

BUILD_ONE(filterforward)
