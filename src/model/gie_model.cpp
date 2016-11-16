//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "gie_model.h"
#include "common/context.h"

GIEModel::GIEModel(const ModelDesc &model_desc, Shape input_shape,
                   int batch_size)
    : Model(model_desc, input_shape, batch_size) {}

void GIEModel::Load() {
  bool use_fp16 = Context::GetContext().GetBool(USEFP16);
  if (use_fp16 && batch_size_ > 1 && batch_size_ % 2 != 0) {
    LOG(FATAL) << "GIE half precision only supports even batch size";
  }

  // FIXME: the input and output blob name is fixed.

  inferer_.reset(new GIEInferer<float>(model_desc_.GetModelDescPath(),
                                       model_desc_.GetModelParamsPath(), "data",
                                       "prob", batch_size_, use_fp16));
  inferer_->CreateEngine();
  input_buffer_ =
      DataBuffer(input_shape_.GetSize() * sizeof(float) * batch_size_);
}

void GIEModel::Forward() {
  DataBuffer output_buffer = DataBuffer(inferer_->GetOutputShape().GetSize() *
                                        sizeof(float) * batch_size_);
  inferer_->DoInference((float *)input_buffer_.GetBuffer(),
                        (float *)output_buffer.GetBuffer());
}

void GIEModel::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  DataBuffer output_buffer = DataBuffer(inferer_->GetOutputShape().GetSize() *
                                        sizeof(float) * batch_size_);
  inferer_->DoInference((float *)input_buffer_.GetBuffer(),
                        (float *)output_buffer.GetBuffer());

  output_shapes_.push_back(inferer_->GetOutputShape());
  output_buffers_.push_back(output_buffer);
}

GIEModel::~GIEModel() { inferer_->DestroyEngine(); }
