//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#ifndef STREAMER_FILE_UTILS_H
#define STREAMER_FILE_UTILS_H
#include "common/common.h"

inline bool FileExists(const string &filename) {
  std::ifstream f(filename);
  return f.good();
}

#endif  // STREAMER_FILE_UTILS_H
