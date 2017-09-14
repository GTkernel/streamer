MACRO(ADD_BUILD_REQS target)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -fopenmp=libomp)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -Wall -Wextra -fopenmp)
  endif()
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
ENDMACRO ()
