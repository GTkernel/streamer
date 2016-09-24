//
// Created by xianran on 9/24/16.
//

#include "model_desc.h"

ModelDesc::ModelDesc(const string &name,
                     const ModelType &type,
                     const string &model_desc_path,
                     const string &model_params_path,
                     int input_width,
                     int input_height)
    : name_(name),
      type_(type),
      model_desc_path_(model_desc_path),
      model_params_path_(model_params_path),
      input_width_(input_width),
      input_height_(input_height) {}

const string &ModelDesc::GetName() const {
  return name_;
}
const ModelType &ModelDesc::GetType() const {
  return type_;
}
const string &ModelDesc::GetModelDescPath() const {
  return model_desc_path_;
}
const string &ModelDesc::GetModelParamsPath() const {
  return model_params_path_;
}
int ModelDesc::GetInputWidth() const {
  return input_width_;
}
int ModelDesc::GetInputHeight() const {
  return input_height_;
}
