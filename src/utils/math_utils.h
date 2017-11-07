
#ifndef STREAMER_UTILS_MATH_UTILS_H_
#define STREAMER_UTILS_MATH_UTILS_H_

inline bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/**
 * @brief Find the largest K numbers for <code>scores</code>.
 * @param scores The list of scores to be compared.
 * @param N The length of scores.
 * @param K Number of numbers to be selected
 * @return
 */
inline std::vector<int> Argmax(float* scores, int N, int K) {
  std::vector<std::pair<float, int>> pairs;
  for (decltype(N) i = 0; i < N; ++i)
    pairs.push_back(std::make_pair(scores[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end(), PairCompare);

  std::vector<int> result;
  for (decltype(K) i = 0; i < K; ++i) result.push_back(pairs[i].second);
  return result;
}

#endif  // STREAMER_UTILS_MATH_UTILS_H_
