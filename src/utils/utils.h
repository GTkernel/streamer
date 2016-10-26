//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef TX1DNN_UTILS_H
#define TX1DNN_UTILS_H

#include "cuda_utils.h"
#include "file_utils.h"
#include "gst_utils.h"
#include "math_utils.h"
#include "string_utils.h"

inline void STREAMER_SLEEP(int secs) {
  std::this_thread::sleep_for(std::chrono::seconds(secs));
}

#endif  // TX1DNN_UTILS_H
