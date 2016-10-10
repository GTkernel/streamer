file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
set(PROJECT_TRDPARTY_DIR "3rdparty")
foreach (SOURCE_FILE ${ALL_SOURCE_FILES})
  string(FIND ${SOURCE_FILE} ${PROJECT_TRDPARTY_DIR} PROJECT_TRDPARTY_DIR_FOUND)
  if (NOT ${PROJECT_TRDPARTY_DIR_FOUND} EQUAL -1)
    list(REMOVE_ITEM ALL_SOURCE_FILES ${SOURCE_FILE})
  endif ()
endforeach ()

message(-style=${PROJECT_SOURCE_DIR}/.clang-format)
add_custom_target(clangformat
  COMMAND
  clang-format
  -style=file
  -i ${ALL_SOURCE_FILES})