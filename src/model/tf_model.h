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

#ifndef STREAMER_MODEL_TF_MODEL_H_
#define STREAMER_MODEL_TF_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <tensorflow/core/public/session.h>
#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "model/model.h"

class TFModel : public Model {
 public:
  TFModel(const ModelDesc& model_desc, Shape input_shape);
  virtual ~TFModel() override;
  virtual void Load() override;
  virtual cv::Mat ConvertAndNormalize(cv::Mat img) override;
  virtual std::unordered_map<std::string, std::vector<cv::Mat>> Evaluate(
      const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map,
      const std::vector<std::string>& output_layer_names) override;

  // do tensorflow model evaluation directly
  virtual std::unordered_map<std::string, std::vector<tensorflow::Tensor>>
  TensorEvaluate(
      const std::vector<std::pair<std::string, tensorflow::Tensor>> inputs,
      const std::vector<std::string>& output_layer_names);

  // start the first layer of model, turn image OpenCV format to Tensor, before
  // evaluation
  virtual std::vector<std::pair<std::string, tensorflow::Tensor>> CV2Tensor(
      const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map);

  // end the last layer of model, back to CV format
  virtual std::unordered_map<std::string, std::vector<cv::Mat>> Tensor2CV(
      const std::unordered_map<std::string, std::vector<tensorflow::Tensor>>&
          input_map);

 private:
  std::unique_ptr<tensorflow::Session> session_;
  std::vector<std::string> layers_;
  std::string input_op_;
  std::string last_op_;
};

#endif  // STREAMER_MODEL_TF_MODEL_H_
