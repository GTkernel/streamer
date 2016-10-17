//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_COMMON_H
#define TX1_DNN_COMMON_H

#include <glog/logging.h>
#include <stdlib.h>
#include <tinytoml/toml.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>
#include "data_buffer.h"
#include "timer.h"
#include "types.h"

using std::string;

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(condition)                                         \
  /* Code block avoids redefinition of cudaError_t error */           \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#endif

//// TOML
toml::Value ParseTomlFromFile(const string &filepath);

#endif  // TX1_DNN_COMMON_H
