// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "processor/neural_net_evaluator.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include "model/model_manager.h"
#include "utils/string_utils.h"
#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

NeuralNetEvaluator::NeuralNetEvaluator(
    const ModelDesc& model_desc, const Shape& input_shape, size_t batch_size,
    const std::vector<std::string>& output_layer_names)
    : Processor(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR, {SOURCE_NAME},
                {SINK_NAME}),
      input_shape_(input_shape),
      batch_size_(batch_size) {
  // Load model.
  auto& manager = ModelManager::GetInstance();
  if (model_desc.GetModelType() == MODEL_TYPE_TENSORFLOW) {
    tf_model_ = std::make_unique<TFModel>(model_desc, input_shape_);
    tf_model_->Load();
  } else {
    model_ = manager.CreateModel(model_desc, input_shape_, batch_size_);
    model_->Load();
  }

  // Create sinks.
  if (output_layer_names.size() == 0) {
    std::string layer = model_desc.GetDefaultOutputLayer();
    if (layer == "") {
      // This case will be triggered if "output_layer_names" is empty and the
      // model's "default_output_layer" parameter was not set. In this case, the
      // NeuralNetEvaluator does not know which layer to treat as the output
      // layer.
      throw std::runtime_error(
          "Unable to create a NeuralNetEvaluator for model \"" +
          model_desc.GetName() + "\" because it does not have a value for " +
          "the \"default_output_layer\" parameter and the NeuralNetEvaluator " +
          "was not constructed with an explicit output layer.");
    }
    LOG(INFO) << "No output layer specified, defaulting to: " << layer;
    PublishLayer(layer);
  } else {
    for (const auto& layer : output_layer_names) {
      PublishLayer(layer);
    }
  }
}

NeuralNetEvaluator::~NeuralNetEvaluator() {
  auto model_raw = model_.release();
  delete model_raw;

  auto tf_model_raw = tf_model_.release();
  delete tf_model_raw;
}

void NeuralNetEvaluator::PublishLayer(std::string layer_name) {
  if (std::find(output_layer_names_.begin(), output_layer_names_.end(),
                layer_name) == output_layer_names_.end()) {
    output_layer_names_.push_back(layer_name);
    LOG(INFO) << "Layer \"" << layer_name << "\" will be published.";
  } else {
    LOG(INFO) << "Layer \"" << layer_name << "\" is already published.";
  }
}

const std::vector<std::string> NeuralNetEvaluator::GetSinkNames() const {
  STREAMER_NOT_IMPLEMENTED;
  return {};
}

std::shared_ptr<NeuralNetEvaluator> NeuralNetEvaluator::Create(
    const FactoryParamsType& params) {
  ModelManager& model_manager = ModelManager::GetInstance();
  std::string model_name = params.at("model");
  CHECK(model_manager.HasModel(model_name));
  ModelDesc model_desc = model_manager.GetModelDesc(model_name);

  size_t num_channels = StringToSizet(params.at("num_channels"));
  Shape input_shape = Shape(num_channels, model_desc.GetInputWidth(),
                            model_desc.GetInputHeight());

  std::vector<std::string> output_layer_names = {
      params.at("output_layer_names")};
  return std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                              output_layer_names);
}

bool NeuralNetEvaluator::Init() { return true; }

bool NeuralNetEvaluator::OnStop() { return true; }

void NeuralNetEvaluator::SetSource(const std::string& name, StreamPtr stream,
                                   const std::string& layername) {
  if (layername == "") {
    if (tf_model_ != NULL) {  // using a tensorflow model
      input_layer_name_ = tf_model_->GetModelDesc().GetDefaultInputLayer();
    } else {
      input_layer_name_ = model_->GetModelDesc().GetDefaultInputLayer();
    }
  } else {
    input_layer_name_ = layername;
  }
  LOG(INFO) << "Using layer \"" << input_layer_name_
            << "\" as input for source \"" << name << "\"";
  Processor::SetSource(name, stream);
}

void NeuralNetEvaluator::SetSource(StreamPtr stream,
                                   const std::string& layername) {
  SetSource(SOURCE_NAME, stream, layername);
}

template <typename T>
void NeuralNetEvaluator::PassFrame(
    std::unordered_map<std::string, std::vector<T>> outputs) {
  // Push the activations for each published layer to their respective sink.
  for (decltype(cur_batch_frames_.size()) i = 0; i < cur_batch_frames_.size();
       ++i) {
    std::unique_ptr<Frame> ret_frame = std::move(cur_batch_frames_.at(i));
    for (const auto& layer_pair : outputs) {
      auto activation_vector = layer_pair.second;
      auto layer_name = layer_pair.first;
      auto activations = activation_vector.at(i);
      ret_frame->SetValue(layer_name, activations);
    }
    //LOG(INFO) << "\n" << ret_frame->ToString();
    ret_frame->SetValue("eval_micros", tf_model_->eval_time);

    PushFrame(SINK_NAME, std::move(ret_frame));
  }
}

StreamPtr NeuralNetEvaluator::GetSink() {
  return Processor::GetSink(SINK_NAME);
}

void NeuralNetEvaluator::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  cv::Mat input_mat;
  std::string frame_name;
  if (input_frame->Count(input_layer_name_) > 0) {
    frame_name = input_layer_name_;
  } else {
    frame_name = "image";
  }

  if (tf_model_ != NULL) {
    // for tensorflow model, only the first layer of model get OpenCV frame at
    // beginning so need to call ConvertAndNormalize
    if (input_layer_name_ == tf_model_->GetModelDesc().GetDefaultInputLayer()) {
      input_mat = input_frame->GetValue<cv::Mat>(frame_name);
      input_frame->SetValue(GetName() + "." + frame_name + ".normalized",
                            tf_model_->ConvertAndNormalize(input_mat));
    }

  } else {
    input_mat = input_frame->GetValue<cv::Mat>(frame_name);
    input_frame->SetValue(GetName() + "." + frame_name + ".normalized",
                          model_->ConvertAndNormalize(input_mat));
  }

  cur_batch_frames_.push_back(std::move(input_frame));
  if (cur_batch_frames_.size() < batch_size_) {
    return;
  }

  std::vector<cv::Mat> cv_batch_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_batch_;

  for (auto& frame : cur_batch_frames_) {
    if (tf_model_ != NULL) {
      if (input_layer_name_ !=
          tf_model_->GetModelDesc().GetDefaultInputLayer()) {
        tensor_batch_.push_back(std::pair<std::string, tensorflow::Tensor>(
            frame_name, frame->GetValue<tensorflow::Tensor>(frame_name)));
        continue;
      }
    }
    cv_batch_.push_back(
        frame->GetValue<cv::Mat>(GetName() + "." + frame_name + ".normalized"));
  }

  if (tf_model_ == NULL) {
    auto layer_outputs =
        model_->Evaluate({{input_layer_name_, cv_batch_}}, output_layer_names_);

    PassFrame(layer_outputs);

  } else {
    // run Tensor Model evaluation
    // check if output_layer_names_ contains last layer
    // if so, need to do Tensor-to-CV transformation
    auto is_last_layer = false;
    if (std::find(output_layer_names_.begin(), output_layer_names_.end(),
                  tf_model_->GetModelDesc().GetDefaultOutputLayer()) !=
        output_layer_names_.end())
      is_last_layer = true;

    std::unordered_map<std::string, std::vector<tensorflow::Tensor>> layer_outputs;

    if (tensor_batch_.size() > 0) {
      // with in the model, pass by Tensor
      layer_outputs =
          tf_model_->TensorEvaluate(tensor_batch_, output_layer_names_);
    } else {
      // at beginning of model, transfer first
      auto tensor_vec_ = tf_model_->CV2Tensor({{input_layer_name_, cv_batch_}});
      layer_outputs =
          tf_model_->TensorEvaluate(tensor_vec_, output_layer_names_);
    }


    if (is_last_layer) {
      auto cv_outputs = tf_model_->Tensor2CV(layer_outputs);
      PassFrame(cv_outputs);
    } else {
      PassFrame(layer_outputs);
    }

  }

  cur_batch_frames_.clear();
}
