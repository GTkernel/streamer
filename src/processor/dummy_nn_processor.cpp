//
// Created by Ran Xian (xranthoar@gmail.com) on 10/21/16.
//

#include "dummy_nn_processor.h"
#include "model/model_manager.h"

DummyNNProcessor::DummyNNProcessor(const ModelDesc &model_desc, int batch_size)
    : Processor({}, 0), model_desc_(model_desc), batch_size_(batch_size) {
  input_shape_ =
      Shape(3, model_desc_.GetInputWidth(), model_desc_.GetInputHeight());
  fake_input_ = DataBuffer(input_shape_.GetSize() * sizeof(float) * batch_size);
}

bool DummyNNProcessor::Init() {
  // Load the model
  model_ = ModelManager::GetInstance().CreateModel(model_desc_, input_shape_,
                                                   batch_size_);
  model_->Load();

  // Prepare fake input
  srand((unsigned)(15213));
  float *data = (float *)fake_input_.GetBuffer();
  for (int i = 0; i < input_shape_.GetSize(); i++) {
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

ProcessorType DummyNNProcessor::GetType() {
  return PROCESSOR_TYPE_DUMMY_NN;
}
