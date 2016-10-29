//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_CUDA_UTILS_H
#define STREAMER_CUDA_UTILS_H

#include "common/common.h"

/**
 * @brief Get a list of GPUs in machine.
 */
inline void GetCUDAGpus(std::vector<int> &gpus) {
  int count = 0;
#ifdef USE_CUDA
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  LOG(FATAL) << "Can't use CUDA function in NO_GPU mode";
#endif
  for (int i = 0; i < count; ++i) {
    gpus.push_back(i);
  }
}

#endif  // STREAMER_CUDA_UTILS_H
