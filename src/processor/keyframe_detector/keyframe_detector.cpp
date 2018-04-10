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

#include "processor/keyframe_detector/keyframe_detector.h"

#include <sstream>

#include "processor/processor.h"
#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME_PREFIX = "output_";

KeyframeDetector::KeyframeDetector(
    const ModelDesc& model_desc, const Shape& shape, const std::string& fv_key,
    std::vector<std::pair<float, size_t>> buf_params)
    : NeuralNetConsumer(PROCESSOR_TYPE_KEYFRAME_DETECTOR, model_desc, shape,
                        {fv_key}, {SOURCE_NAME}, {}) {
  Processor::SetSource(SOURCE_NAME, nne_->GetSink());
  Setup(fv_key, buf_params);
}

KeyframeDetector::KeyframeDetector(
    const std::string& fv_key, std::vector<std::pair<float, size_t>> buf_params)
    : NeuralNetConsumer(PROCESSOR_TYPE_KEYFRAME_DETECTOR, {SOURCE_NAME}, {}) {
  Setup(fv_key, buf_params);
}

std::shared_ptr<KeyframeDetector> KeyframeDetector::Create(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void KeyframeDetector::EnableLog(std::string output_dir) {
  for (decltype(bufs_.size()) i = 0; i < bufs_.size(); i++) {
    bufs_.at(i)->EnableLog(output_dir, "keyframe_buffer_" + std::to_string(i));
  }
}

void KeyframeDetector::SetSelectivity(size_t buf_idx, float new_sel) {
  bufs_.at(buf_idx)->SetSelectivity(new_sel);
}

std::string KeyframeDetector::GetSinkName(size_t buf_idx) {
  return SINK_NAME_PREFIX + std::to_string(buf_idx);
}

void KeyframeDetector::SetSource(StreamPtr stream) {
  NeuralNetConsumer::SetSource(SOURCE_NAME, stream);
}

bool KeyframeDetector::OnStop() {
  for (const auto& buf : bufs_) {
    buf->Stop();
  }
  return NeuralNetConsumer::OnStop();
}

void KeyframeDetector::Process() {
  std::vector<std::unique_ptr<Frame>> frames;
  frames.push_back(GetFrame(SOURCE_NAME));

  // For every level in the hierarchy...
  for (decltype(bufs_.size()) i = 0; i < bufs_.size(); ++i) {
    std::vector<std::unique_ptr<Frame>> keyframes;

    // For every new frame that should be added to this level...
    for (auto& frame : frames) {
      auto start_time_micros = boost::posix_time::microsec_clock::local_time();
      auto new_keyframes = bufs_.at(i)->Push(std::move(frame));
      auto kd_micros =
          boost::posix_time::microsec_clock::local_time() - start_time_micros;

      bool recorded_time = false;
      // Accumulate the keyframes created by adding this frame to the level.
      for (auto& keyframe : new_keyframes) {
        if (!recorded_time) {
          std::ostringstream time_key;
          time_key << "kd_level_" << i << "_micros";
          keyframe->SetValue(time_key.str(), kd_micros);
          recorded_time = true;
        }

        keyframes.push_back(std::move(keyframe));
      }
    }

    // Push all of the new keyframes from this level to the appropriate sink.
    for (auto& keyframe : keyframes) {
      // We need to copy the frame because we still need to keep a copy in
      // "keyframes".
      PushFrame(GetSinkName(i), std::make_unique<Frame>(keyframe));
    }

    // In the next iteration, all of the keyframes from this level will be
    // pushed to the next level.
    frames = std::move(keyframes);
  }
}

void KeyframeDetector::Setup(const std::string& fv_key,
                             std::vector<std::pair<float, size_t>> buf_params) {
  CHECK(buf_params.size() > 0)
      << "Unable to create a KeyframeDetector with a 0-level hierarchy.";
  for (decltype(buf_params.size()) i = 0; i < buf_params.size(); ++i) {
    std::pair<float, size_t> params = buf_params.at(i);
    float sel = params.first;
    size_t buf_len = params.second;
    bufs_.push_back(std::make_unique<KeyframeBuffer>(fv_key, sel, buf_len, i));
    sinks_.insert({GetSinkName(i), StreamPtr(new Stream())});
  }
}
