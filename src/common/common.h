//
// Created by Ran Xian on 7/22/16.
//

#ifndef STREAMER_COMMON_COMMON_H_
#define STREAMER_COMMON_COMMON_H_

#include <glog/logging.h>
#include <stdlib.h>
#include <tinytoml/include/toml/toml.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "timer.h"
#include "types.h"

using std::string;

#define STREAMER_NOT_IMPLEMENTED (LOG(FATAL) << "Function not implemented");

#endif  // STREAMER_COMMON_COMMON_H_
