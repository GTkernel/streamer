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

#ifndef STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_
#define STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_

#include <unordered_map>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

#include "model/tf_model.h"
// A NeuralNetEvaluator is a Processor that runs deep neural network inference.
// This Processor has only one source, named "input". On creation, the
// higher-level code specifies which layers of the DNN should be published. One
// sink is created for each published layer and is named after the layer.
// At any time, PublishLayer() can be called to expose a previously unpublished
// layer.
class NeuralNetEvaluator : public Processor {
 public:
  // If output_layer_names is empty, then by default the last layer is
  // published.
  NeuralNetEvaluator(const ModelDesc& model_desc, const Shape& input_shape,
                     size_t batch_size = 1,
                     const std::vector<std::string>& output_layer_names = {});
  ~NeuralNetEvaluator();

  // Adds layer_name to the list of the layers whose activations will be
  // published.
  void PublishLayer(std::string layer_name);
  // Returns a vector of the names of this NeuralNetEvaluator's sinks, which are
  // the names of the layers that it is publishing.
  const std::vector<std::string> GetSinkNames() const;

  static std::shared_ptr<NeuralNetEvaluator> Create(
      const FactoryParamsType& params);

  // Hides Processor::SetSource(const std::string&, StreamPtr)
  void SetSource(const std::string& name, StreamPtr stream,
                 const std::string& layername = "");
  void SetSource(StreamPtr stream, const std::string& layername = "");
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // Executes the neural network and returns a mapping from the name of a layer
  // to that layer's activations.
  template <typename T>
  void PassFrame(std::unordered_map<std::string, std::vector<T>> outputs);
  std::unordered_map<std::string, cv::Mat> Evaluate();

  Shape input_shape_;
  std::string input_layer_name_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<TFModel> tf_model_;
  std::vector<std::string> output_layer_names_;
  std::vector<std::unique_ptr<Frame>> cur_batch_frames_;
  size_t batch_size_;
};

#endif  // STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_
