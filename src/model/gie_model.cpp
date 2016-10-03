//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "gie_model.h"

GIEModel::GIEModel(const ModelDesc &model_desc, Shape input_shape)
    : Model(model_desc, input_shape) {}

void GIEModel::Load() {
  // FIXME: the input and output blob name is fixed.
  inferer_.reset(new GIEInferer<DType>(model_desc_.GetModelDescPath(),
                                       model_desc_.GetModelParamsPath(), "data",
                                       "prob"));
  inferer_->CreateEngine();
  input_buffer_ = DataBuffer(input_shape_.GetSize() * sizeof(float));
}

void GIEModel::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  DataBuffer output_buffer =
      DataBuffer(inferer_->GetOutputShape().GetSize() * sizeof(float));
  inferer_->DoInference((DType *)input_buffer_.GetBuffer(),
                        (DType *)output_buffer.GetBuffer());

  output_shapes_.push_back(inferer_->GetOutputShape());
  output_buffers_.push_back(output_buffer);
}

GIEModel::~GIEModel() { inferer_->DestroyEngine(); }
