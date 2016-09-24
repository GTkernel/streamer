file(GLOB APP_SRCS ${PROJECT_SOURCE_DIR}/apps/*.cpp)

foreach(f ${APP_SRCS})
  get_filename_component(basename ${f} NAME_WE)
  add_executable(${basename} ${f})
  target_link_libraries(${basename} tx1dnn)
  install(TARGETS ${basename} DESTINATION apps)
endforeach()
