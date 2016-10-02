//
// Created by Ran Xian on 7/29/16.
//

#ifndef TX1DNN_UTILS_H
#define TX1DNN_UTILS_H

#include "common.h"

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

inline bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}

/**
 * @brief Find the largest K numbers for <code>scores</code>.
 * @param scores The list of scores to be compared.
 * @param N The length of scores.
 * @param K Number of numbers to be selected
 * @return
 */
inline std::vector<int> Argmax(float *scores, int N, int K) {
  std::vector <std::pair<float, int>> pairs;
  for (size_t i = 0; i < N; ++i)
    pairs.push_back(std::make_pair(scores[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < K; ++i)
    result.push_back(pairs[i].second);
  return result;
}

#endif //TX1DNN_UTILS_H
