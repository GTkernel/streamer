add_executable(tx1_run_alexnet tx1_run_alexnet.cpp)
target_link_libraries(tx1_run_alexnet tx1dnn)

install(TARGETS tx1_run_alexnet DESTINATION apps)

add_executable(tx1_classify tx1_classify.cpp)
target_link_libraries(tx1_classify tx1dnn)

install(TARGETS tx1_classify DESTINATION apps)

