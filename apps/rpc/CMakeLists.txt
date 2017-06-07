if (NOT USE_RPC)
  return ()
endif ()

MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_dependencies(apps ${app})
ENDMACRO()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

add_executable(rpc_sender EXCLUDE_FROM_ALL rpc_sender.cpp)
BUILD_ONE(rpc_sender)

add_executable(rpc_receiver EXCLUDE_FROM_ALL rpc_receiver.cpp)
BUILD_ONE(rpc_receiver)
