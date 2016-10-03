//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "caffe_fp16_model.h"

CaffeFp16Model::CaffeFp16Model(const ModelDesc &model_desc, Shape input_shape)
    : Model(model_desc, input_shape) {}

void CaffeFp16Model::Load() {
  // Load the network.
  net_.reset(new caffe::Net<DType, MType>(model_desc_.GetModelDescPath(),
                                          caffe::TEST));
  net_->CopyTrainedLayersFrom(model_desc_.GetModelParamsPath());

  CHECK(input_shape_.channel == 3 || input_shape_.channel == 1)
  << "Input layer should have 1 or 3 channels.";

  caffe::Blob<DType, MType> *input_layer = net_->input_blobs()[0];

  // Adjust input dimensions
  input_layer->Reshape(1,
                       input_shape_.channel,
                       input_shape_.height,
                       input_shape_.width);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Prepare input buffer
  input_buffer_ = DataBuffer(input_shape_.GetSize() * sizeof(float));
}

void CaffeFp16Model::Evaluate() {
  // Copy the input to half bit input
  if (sizeof(DType) == 2) {
    float *fp32data = (float *) input_buffer_.GetBuffer();
    caffe::Blob<DType, MType> *input_layer = net_->input_blobs()[0];
    DType *fp16data = (DType *) (input_layer->mutable_cpu_data());

    size_t image_size = input_shape_.GetSize();
    for (size_t i = 0; i < image_size; i++) {
      fp16data[i] = caffe::Get<DType>(fp32data[i]);
    }
  }

  // Evaluate
  output_shapes_.clear();
  output_buffers_.clear();

  net_->ForwardPrefilled();

  // Copy the output of the network
  auto output_blobs = net_->output_blobs();
  for (auto output_blob : output_blobs) {
    Shape shape
        (output_blob->channels(), output_blob->width(), output_blob->height());
    DataBuffer output_buffer;
    if (sizeof(DType) == 2) {
      output_buffer = DataBuffer(shape.GetSize() * sizeof(float));
      float *fp32data = (float *) output_buffer.GetBuffer();
      DType *fp16data = output_blob->mutable_cpu_data();
      size_t len = shape.GetSize();
      for (size_t i = 0; i < len; i++) {
        fp32data[i] = caffe::Get<float>(fp16data[i]);
      }
    } else {
      output_buffer = DataBuffer(output_blob->mutable_cpu_data(),
                                 shape.GetSize() * sizeof(float));
    }

    output_shapes_.push_back(shape);
    output_buffers_.push_back(output_buffer);
  }
}
