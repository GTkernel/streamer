//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model.h"
#include <string>
#include <vector>
#include "common/types.h"

Model::Model(const ModelDesc& model_desc, Shape input_shape, size_t batch_size)
    : model_desc_(model_desc),
      input_shape_(input_shape),
      batch_size_(batch_size) {}

Model::~Model() {}

ModelDesc Model::GetModelDesc() const { return model_desc_; }

cv::Mat Model::ConvertAndNormalize(cv::Mat img) {
  return img;
}
