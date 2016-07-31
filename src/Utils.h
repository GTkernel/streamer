//
// Created by Ran Xian on 7/29/16.
//

#ifndef TX1DNN_UTILS_H
#define TX1DNN_UTILS_H

#include "common.h"
#include <caffe/caffe.hpp>

void get_gpus(std::vector<int> &gpus);

bool PairCompare(const std::pair<float, int>& lhs,
                 const std::pair<float, int>& rhs);

std::vector<int> Argmax(const std::vector<float>& v, int N);

#endif //TX1DNN_UTILS_H
