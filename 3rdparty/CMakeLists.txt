if (TEST_ON)
  add_subdirectory(googletest)
endif ()
add_subdirectory(struck)

# linenoise
add_library(linenoise linenoise/linenoise.c linenoise/linenoise.h)
