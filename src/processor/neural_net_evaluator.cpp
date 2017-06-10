
#include "processor/neural_net_evaluator.h"
#include "model/model_manager.h"

constexpr auto SOURCE_NAME = "input";

NeuralNetEvaluator::NeuralNetEvaluator(
    const ModelDesc& model_desc, const Shape& input_shape,
    const std::vector<std::string>& output_layer_names)
    : Processor(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR, {SOURCE_NAME}, {}),
      input_shape_(input_shape) {
  // Load model.
  auto& manager = ModelManager::GetInstance();
  model_ = manager.CreateModel(model_desc, input_shape_, 1);
  model_->Load();

  // Create sinks.
  if (output_layer_names.size() == 0) {
    std::string layer = *(model_->GetLayerNames().end() - 1);
    LOG(INFO) << "No output layer specified, defaulting to: " << layer;
    PublishLayer(layer);
  } else {
    for (const auto& layer : output_layer_names) {
      PublishLayer(layer);
    }
  }

  // Prepare data buffer.
  input_buffer_ = DataBuffer(input_shape_.GetSize() * sizeof(float));
}

void NeuralNetEvaluator::PublishLayer(std::string layer_name) {
  if (sinks_.find(layer_name) == sinks_.end()) {
    sinks_.insert({layer_name, std::make_shared<Stream>(layer_name)});
  } else {
    LOG(INFO) << "Layer \"" << layer_name << "\" is already published.";
  }
}

const std::vector<std::string> NeuralNetEvaluator::GetSinkNames() const {
  std::vector<std::string> sink_names;
  for (const auto& sink_pair : sinks_) {
    sink_names.push_back(sink_pair.first);
  }
  return sink_names;
}

bool NeuralNetEvaluator::Init() { return true; }

bool NeuralNetEvaluator::OnStop() { return true; }

void NeuralNetEvaluator::Process() {
  // Prepare the image for inference by splitting its channels.
  float* data = (float*)input_buffer_.GetBuffer();
  auto image_frame = GetFrame<ImageFrame>(SOURCE_NAME);
  cv::Mat img = image_frame->GetImage();
  CHECK(img.channels() == input_shape_.channel &&
        img.size[0] == input_shape_.width &&
        img.size[1] == input_shape_.height);
  std::vector<cv::Mat> output_channels;
  // This loop creates a cv::Mat for each channel that is configured to point to
  // a particular location in "data", but the data itself is not populated until
  // the call to cv::split().
  for (decltype(input_shape_.channel) j = 0; j < input_shape_.channel; ++j) {
    cv::Mat channel(input_shape_.height, input_shape_.width, CV_32F, data);
    output_channels.push_back(channel);
    data += input_shape_.width * input_shape_.height;
  }
  cv::split(img, output_channels);

  auto layer_outputs = Evaluate();

  // Push the activations for each published layer to their respective sink.
  for (const auto& layer_pair : layer_outputs) {
    // TODO: Populate the layer_name parameter.
    auto layer_frame =
        new LayerFrame("", layer_pair.second, image_frame->GetOriginalImage());
    PushFrame(layer_pair.first, layer_frame);
  }
}

std::unordered_map<std::string, cv::Mat> NeuralNetEvaluator::Evaluate() {
  auto input_buffer = model_->GetInputBuffer();
  input_buffer.Clone(input_buffer_);

  model_->Evaluate();

  std::unordered_map<std::string, cv::Mat> layer_to_acts;
  for (const auto& sink_pair : sinks_) {
    layer_to_acts[sink_pair.first] = model_->GetLayerOutput(sink_pair.first);
  }
  return layer_to_acts;
}
