//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model.h"

Model::Model(const ModelDesc& model_desc, Shape input_shape, size_t batch_size)
    : model_desc_(model_desc),
      input_shape_(input_shape),
      batch_size_(batch_size) {}

Model::~Model() {}

ModelDesc Model::GetModelDesc() const { return model_desc_; }
std::unordered_map<std::string, std::vector<cv::Mat>> Model::Evaluate(
    cv::Mat input) {
  return Evaluate({{model_desc_.GetDefaultInputLayer(), {input}}},
                  {model_desc_.GetDefaultOutputLayer()});
}
