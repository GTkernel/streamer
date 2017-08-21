
execute_process(COMMAND git submodule update --init
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if (TEST_ON)
  add_subdirectory(googletest)
endif ()

if (USE_DLIB)
  add_subdirectory(dlib)
endif()

# litesql database library
add_subdirectory(litesql)
