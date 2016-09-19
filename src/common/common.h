//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_COMMON_H
#define TX1_DNN_COMMON_H

#include <opencv2/opencv.hpp>
#include <string>
#include <glog/logging.h>
#include <thread>
#include <memory>

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#include "timer.h"
#include "types.h"

using std::string;

#endif //TX1_DNN_COMMON_H
