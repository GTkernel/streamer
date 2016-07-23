add_executable(tx1_run_alexnet tx1_run_alexnet.cpp)
target_link_libraries(tx1_run_alexnet tx1dnn ${CUDA_LIBRARIES})

install(TARGETS tx1_run_alexnet DESTINATION apps)
