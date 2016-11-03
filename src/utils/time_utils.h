//
// Created by Ran Xian (xranthoar@gmail.com) on 11/2/16.
//

#ifndef STREAMER_TIME_UTILS_H
#define STREAMER_TIME_UTILS_H

#include <ctime>
#include "common/common.h"

inline string GetCurrentTimeString(const string &time_format) {
  std::time_t t = std::time(nullptr);
  char timestr[128];

  std::strftime(timestr, sizeof(timestr), time_format.c_str(),
                std::localtime(&t));

  return string(timestr);
}

#endif  // STREAMER_TIME_UTILS_H
