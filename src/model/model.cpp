//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model.h"

Model::Model(const ModelDesc &model_desc, Shape input_shape)
    : model_desc_(model_desc), input_shape_(input_shape) {}

ModelDesc Model::GetModelDesc() const {
  return model_desc_;
}
DataBuffer Model::GetInputBuffer() {
  return input_buffer_;
}
std::vector<DataBuffer> Model::GetOutputBuffers() {
  return output_buffers_;
}
std::vector<Shape> Model::GetOutputShapes() {
  return output_shapes_;
}
