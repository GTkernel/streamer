
execute_process(COMMAND git submodule update --init
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if (TEST_ON)
  add_subdirectory(googletest)
endif ()

if (USE_STRUCK)
  add_subdirectory(struck)
endif ()

# litesql database library
add_subdirectory(litesql)
