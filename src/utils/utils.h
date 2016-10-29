//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_UTILS_H
#define STREAMER_UTILS_H

#include "cuda_utils.h"
#include "file_utils.h"
#include "gst_utils.h"
#include "math_utils.h"
#include "string_utils.h"

inline void STREAMER_SLEEP(int msecs) {
  std::this_thread::sleep_for(std::chrono::milliseconds(msecs));
}

#endif  // STREAMER_UTILS_H
