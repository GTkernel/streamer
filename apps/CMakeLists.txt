find_package(Threads REQUIRED)
find_package(Boost COMPONENTS regex system thread filesystem date_time REQUIRED)

# Macro to get sub directories
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

MACRO(BUILD_APP app_dir)
  get_filename_component(basename ${app_dir} NAME_WE)
  FILE(GLOB app_srcs ${app_dir}/*.cpp ${app_dir}/*.h ${app_dir}/*.c)
  add_executable(${basename} ${app_srcs})
  if (JEMALLOC_FOUND)
    target_link_libraries(${basename} ${JAMALLOC_LIBRARIES})
  endif ()
  target_link_libraries(${basename} streamer ${Boost_PROGRAM_OPTIONS_LIBRARY})
  install(TARGETS ${basename} DESTINATION apps)
ENDMACRO()

set(APPS_ROOT ${PROJECT_SOURCE_DIR}/apps/)
SUBDIRLIST(APP_DIRS ${APPS_ROOT})

set(PTGRAY_APP_DIRS gige)
list(REMOVE_ITEM APP_DIRS ${PTGRAY_APP_DIRS})

foreach (app_dir ${APP_DIRS})
  BUILD_APP(${app_dir})
endforeach ()

if (USE_PTGRAY)
  foreach (app_dir ${PTGRAY_APP_DIRS})
    BUILD_APP(${app_dir})
  endforeach ()
  target_link_libraries(gige ${Boost_FILESYSTEM_LIBRARY})
endif ()

target_link_libraries(server ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
