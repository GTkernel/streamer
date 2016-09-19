add_executable(tx1_classify tx1_classify.cpp)

target_link_libraries(tx1_classify tx1dnn)

install(TARGETS tx1_classify DESTINATION apps)

