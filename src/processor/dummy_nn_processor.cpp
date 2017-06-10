//
// Created by Ran Xian (xranthoar@gmail.com) on 10/21/16.
//

#include "dummy_nn_processor.h"

#include "common/types.h"
#include "model/model_manager.h"

DummyNNProcessor::DummyNNProcessor(const ModelDesc& model_desc)
    : Processor(PROCESSOR_TYPE_DUMMY_NN, {}, {}), model_desc_(model_desc) {
  input_shape_ =
      Shape(3, model_desc_.GetInputWidth(), model_desc_.GetInputHeight());
  fake_input_ = DataBuffer(input_shape_.GetSize() * sizeof(float) * 1);
}

bool DummyNNProcessor::Init() {
  // Load the model
  model_ =
      ModelManager::GetInstance().CreateModel(model_desc_, input_shape_, 1);
  model_->Load();

  // Prepare fake input
  srand((unsigned)(15213));
  float* data = (float*)fake_input_.GetBuffer();
  for (decltype(input_shape_.GetSize()) i = 0; i < input_shape_.GetSize();
       ++i) {
    data[i] = (float)(rand()) / (float)(RAND_MAX);
  }

  // Copy the fake input
  DataBuffer input_buffer = model_->GetInputBuffer();
  input_buffer.Clone(fake_input_);

  return true;
}

bool DummyNNProcessor::OnStop() {
  this->model_ = nullptr;
  return true;
}

void DummyNNProcessor::Process() { model_->Forward(); }
