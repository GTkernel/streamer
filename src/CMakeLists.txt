set(TX1DNN_SOURCE_FILES GstVideoCapture.cpp)

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})