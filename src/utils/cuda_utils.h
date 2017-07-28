//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_UTILS_CUDA_UTILS_H_
#define STREAMER_UTILS_CUDA_UTILS_H_

#include "common/common.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(condition)                                         \
  /* Code block avoids redefinition of cudaError_t error */           \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#endif  // USE_CUDA

/**
 * @brief Get a list of GPUs in machine.
 */
inline void GetCUDAGpus(std::vector<int>& gpus) {
  int count = 0;
#ifdef USE_CUDA
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  LOG(FATAL) << "Can't use CUDA function in NO_GPU mode";
#endif  // USE_CUDA
  for (int i = 0; i < count; ++i) {
    gpus.push_back(i);
  }
}

#endif  // STREAMER_UTILS_CUDA_UTILS_H_
