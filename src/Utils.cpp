//
// Created by Ran Xian on 7/29/16.
//

#include "Utils.h"

void get_gpus(std::vector<int> &gpus) {
  int count = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  NO_GPU;
#endif
  for (int i = 0; i < count; ++i) {
    gpus.push_back(i);
  }
}

bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}