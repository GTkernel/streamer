
#include "processor/keyframe_detector/keyframe_detector.h"

#include <sstream>

#include "processor/processor.h"
#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME_PREFIX = "output_";

KeyframeDetector::KeyframeDetector(
    const ModelDesc& model_desc, const Shape& shape, std::string layer,
    std::vector<std::pair<float, size_t>> buf_params)
    : NeuralNetConsumer(PROCESSOR_TYPE_KEYFRAME_DETECTOR, model_desc, shape,
                        {layer}, {SOURCE_NAME}, {}) {
  std::string nne_sink_name = nne_->GetSinkNames().at(0);
  Processor::SetSource(SOURCE_NAME, nne_->GetSink(nne_sink_name));
  Setup(buf_params);
}

KeyframeDetector::KeyframeDetector(
    std::vector<std::pair<float, size_t>> buf_params)
    : NeuralNetConsumer(PROCESSOR_TYPE_KEYFRAME_DETECTOR, {SOURCE_NAME}, {}) {
  Setup(buf_params);
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
      // Accumulate the keyframes created by adding this frame to the level.
      for (auto& keyframe : bufs_.at(i)->Push(std::move(frame))) {
        keyframes.push_back(std::move(keyframe));
      }
    }

    std::ostringstream msg;
    msg << "Keyframe detector level " << i << " found keyframes: { ";
    // Push all of the new keyframes from this level to the appropriate sink.
    for (auto& keyframe : keyframes) {
      msg << keyframe->GetValue<unsigned long>("frame_id") << " ";
      // We need to copy the frame because we still need to keep a copy in
      // "keyframes".
      PushFrame(GetSinkName(i), std::make_unique<Frame>(keyframe));
    }
    msg << "}";
    if (keyframes.size()) {
      // If we detected any keyframes, then print their ids.
      LOG(INFO) << msg.str();
    }

    // In the next iteration, all of the keyframes from this level will be
    // pushed to the next level.
    frames = std::move(keyframes);
  }
}

void KeyframeDetector::Setup(std::vector<std::pair<float, size_t>> buf_params) {
  CHECK(buf_params.size() > 0)
      << "Unable to create a KeyframeDetector with a 0-level hierarchy.";
  for (decltype(buf_params.size()) i = 0; i < buf_params.size(); ++i) {
    std::pair<float, size_t> params = buf_params.at(i);
    float sel = params.first;
    size_t buf_len = params.second;
    bufs_.push_back(std::make_unique<KeyframeBuffer>(sel, buf_len));
    sinks_.insert({GetSinkName(i), StreamPtr(new Stream())});
  }
}

std::string KeyframeDetector::GetSinkName(size_t buf_idx) {
  return SINK_NAME_PREFIX + std::to_string(buf_idx);
}
