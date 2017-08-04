MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

add_executable(publish EXCLUDE_FROM_ALL publish.cpp)
BUILD_ONE(publish)

add_executable(subscribe EXCLUDE_FROM_ALL subscribe.cpp)
BUILD_ONE(subscribe)
