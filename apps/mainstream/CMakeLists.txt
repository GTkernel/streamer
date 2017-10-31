MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

add_executable(mainstream_server EXCLUDE_FROM_ALL mainstream_server.cpp)
BUILD_ONE(mainstream_server)

add_executable(mainstream_performance EXCLUDE_FROM_ALL mainstream_performance.cpp)
BUILD_ONE(mainstream_performance)
