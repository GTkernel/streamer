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
  "${CMAKE_CURRENT_BINARY_DIR}/framesdatabase.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/framesdatabase.hpp"
  )

add_custom_command(
  OUTPUT ${framesdb_sources}
  COMMAND litesql-gen -t c++ --output-dir=${CMAKE_CURRENT_BINARY_DIR} framesdb.xml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(train_detector_publisher EXCLUDE_FROM_ALL
  train_detector_publisher.cpp)

add_executable(train_detector_subscriber EXCLUDE_FROM_ALL
  train_detector_subscriber.cpp
  db_filewriter.cpp
  db_filewriter.h
  ${framesdb_sources})

BUILD_ONE(train_detector_publisher)
BUILD_ONE(train_detector_subscriber)
