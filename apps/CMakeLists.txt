add_executable(tx1_classify tx1_classify.cpp)

if (USE_CAFFE)
  target_link_libraries(tx1_classify tx1dnn-caffe)
endif ()

if (USE_GIE)
  target_link_libraries(tx1_classify tx1dnn-gie)
endif ()

if (USE_MXNET)
  target_link_libraries(tx1_classify tx1dnn-mxnet)
endif ()

install(TARGETS tx1_classify DESTINATION apps)

