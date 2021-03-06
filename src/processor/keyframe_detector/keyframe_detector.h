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

#ifndef STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_DETECTOR_H_
#define STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_DETECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/types.h"
#include "model/model.h"
#include "processor/keyframe_detector/keyframe_buffer.h"
#include "processor/neural_net_consumer.h"

// This Processor decimates the incoming frame stream by only forwarding
// keyframes, which are extracted at multiple granularities. The sinks are named
// "output_<i>", where "i" is the index of a particular granularity. The
// different granularies and their sinks are ordered corresponding to the order
// of the "buf_params" constructor parameter. The selectivity and buffer length
// for each granularity are controlled independently.
class KeyframeDetector : public NeuralNetConsumer {
 public:
  // This constructor automatically creates a hidden NeuralNetworkEvaluator.
  // "buf_params" is a vector of pairs where the first element is a selectivity
  // in the range (0, 1], and the second element is a buffer length.
  KeyframeDetector(const ModelDesc& model_desc, const Shape& shape,
                   const std::string& fv_key,
                   std::vector<std::pair<float, size_t>> buf_params);
  // This constructor relies on the calling code to connece this
  // KeyframeDetector's source to a preexisting NeuralNetworkEvaluator's sink.
  KeyframeDetector(const std::string& fv_key,
                   std::vector<std::pair<float, size_t>> buf_params);

  static std::shared_ptr<KeyframeDetector> Create(
      const FactoryParamsType& params);
  // Signals the KeyframeDetector to begin logging the "frame_id" field of each
  // keyframe to a hierarchy-specific file. See "KeyframeBuffer::EnableLog()".
  void EnableLog(std::string output_dir);
  // "new_sel" must be in the range (0, 1].
  void SetSelectivity(size_t buf_idx, float new_selelectivity);
  std::string GetSinkName(size_t buf_idx);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  void Setup(const std::string& fv_key,
             std::vector<std::pair<float, size_t>> buf_params);

  // The entries detect keyframes at progressively coarser granularities.
  std::vector<std::unique_ptr<KeyframeBuffer>> bufs_;
};

#endif  // STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_DETECTOR_H_
