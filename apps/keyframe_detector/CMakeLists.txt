MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

add_executable(keyframe_detector EXCLUDE_FROM_ALL keyframe_detector.cpp)
BUILD_ONE(keyframe_detector)

add_executable(vary_kd_params EXCLUDE_FROM_ALL vary_kd_params.cpp)
BUILD_ONE(vary_kd_params)
