//
// Created by Ran Xian (xranthoar@gmail.com) on 9/25/16.
//

#ifndef TX1DNN_STRINGUTILS_H
#define TX1DNN_STRINGUTILS_H
#include "common/common.h"

/**
 * @brief Determine if a string ends with certain suffix.
 *
 * @param str The string to check.
 * @param ending The suffix.
 *
 * @return True if the string ends with ending.
 */
inline bool EndsWith(const string &str, const string &ending) {
  if (ending.size() > str.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), str.rbegin());
}

/**
 * @brief Determine if a string ends with certain prefix.
 *
 * @param str The string to check.
 * @param ending The prefix.
 *
 * @return True if the string starts with <code>start</code>.
 */
inline bool StartsWith(const string &str, const string &start) {
  if (start.size() > str.size()) return false;
  return std::equal(start.begin(), start.end(), str.begin());
}

inline string TrimSpaces(const string &str) {
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

#endif //TX1DNN_STRINGUTILS_H
