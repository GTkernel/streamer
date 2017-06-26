if (TEST_ON)
  add_subdirectory(googletest)
endif ()

# linenoise
add_library(linenoise linenoise/linenoise.c linenoise/linenoise.h)

# json
add_subdirectory(json)

# litesql database library
add_subdirectory(litesql)
