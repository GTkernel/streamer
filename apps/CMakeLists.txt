# Macro to get sub directories

# find_package(Boost)

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH (child ${children})
    IF (IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF ()
  ENDFOREACH ()
  SET(${result} ${dirlist})
ENDMACRO()

set(APPS_ROOT ${PROJECT_SOURCE_DIR}/apps/)
SUBDIRLIST(APP_DIRS ${APPS_ROOT})

foreach (app_dir ${APP_DIRS})
  get_filename_component(basename ${app_dir} NAME_WE)
  FILE(GLOB app_srcs ${app_dir}/*.cpp ${app_dir}/*.h ${app_dir}/*.c)
  add_executable(${basename} ${app_srcs})
  if (JEMALLOC_FOUND)
    target_link_libraries(${basename} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${basename} tx1dnn ${Boost_PROGRAM_OPTIONS_LIBRARY})
  install(TARGETS ${basename} DESTINATION apps)
endforeach ()

target_link_libraries(runner linenoise)
