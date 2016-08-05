if (USE_GIE)
	set(TX1DNN_LIBRARIES tx1dnn-gie)
else()
	set(TX1DNN_LIBRARIES tx1dnn-caffe)
endif()

add_executable(tx1_run_alexnet tx1_run_alexnet.cpp)
target_link_libraries(tx1_run_alexnet ${TX1DNN_LIBRARIES} tx1dnn-mxnet)

install(TARGETS tx1_run_alexnet DESTINATION apps)

add_executable(tx1_classify tx1_classify.cpp)
target_link_libraries(tx1_classify ${TX1DNN_LIBRARIES} tx1dnn-mxnet)

install(TARGETS tx1_classify DESTINATION apps)

