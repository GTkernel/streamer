//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_COMMON_H
#define TX1_DNN_COMMON_H

#include <opencv2/opencv.hpp>
#include <string>
#include <glog/logging.h>
#include <thread>
#include <memory>
#include <caffe/util/float16.hpp>
#include "Timer.h"
#include "Type.h"

using std::string;

#ifdef ON_MAC
#define float16 float
#define CAFFE_FP16_MTYPE float
#else
using caffe::float16;
#endif


#endif //TX1_DNN_COMMON_H
