MACRO(BUILD_ONE app)
  if (JEMALLOC_FOUND)
    target_link_libraries(${app} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${app} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  target_link_libraries(${app} litesql)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/litesql/include)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(apps ${app})
  ADD_BUILD_REQS(${app})
ENDMACRO()

# Generate stubs for litesql database library
set(framesdb_sources
  "${CMAKE_CURRENT_BINARY_DIR}/framesdb.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/framesdb.hpp"
  )

add_custom_command(
  OUTPUT ${framesdb_sources}
  COMMAND litesql-gen -t c++
    --output-dir=${CMAKE_CURRENT_BINARY_DIR} framesdb.xml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_source_files_properties(${framesdb_sources}
  PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)

add_executable(train_detector EXCLUDE_FROM_ALL
  train_detector.cpp
  litesql_writer.cpp
  litesql_writer.h
  ${framesdb_sources})

BUILD_ONE(train_detector)
