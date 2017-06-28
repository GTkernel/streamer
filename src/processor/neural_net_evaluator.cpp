
#include "neural_net_evaluator.h"

#include "model/model_manager.h"
#include "utils/string_utils.h"

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
    std::string layer = model_desc.GetLastLayer();
    LOG(INFO) << "No output layer specified, defaulting to: " << layer;
    PublishLayer(layer);
  } else {
    for (const auto& layer : output_layer_names) {
      PublishLayer(layer);
    }
  }
}

void NeuralNetEvaluator::PublishLayer(std::string layer_name) {
  if (sinks_.find(layer_name) == sinks_.end()) {
    sinks_.insert({layer_name, std::make_shared<Stream>(layer_name)});
    LOG(INFO) << "Layer \"" << layer_name << "\" will be published.";
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
  return std::make_shared<NeuralNetEvaluator>(model_desc, input_shape,
                                              output_layer_names);
}

bool NeuralNetEvaluator::Init() { return true; }

bool NeuralNetEvaluator::OnStop() { return true; }

void NeuralNetEvaluator::Process() {
  auto image_frame = GetFrame(SOURCE_NAME);
  cv::Mat img = image_frame->GetValue<cv::Mat>("Image");
  CHECK(img.channels() == input_shape_.channel &&
        img.size[0] == input_shape_.width &&
        img.size[1] == input_shape_.height);

  std::vector<std::string> output_layer_names;
  for (const auto& sink_pair : sinks_) {
    output_layer_names.push_back(sink_pair.first);
  }

  auto layer_outputs = model_->Evaluate(img, output_layer_names);

  // Push the activations for each published layer to their respective sink.
  for (const auto& layer_pair : layer_outputs) {
    auto layer_frame = std::make_unique<Frame>(image_frame);
    layer_frame->SetValue("Activations", layer_pair.second);
    layer_frame->SetValue("LayerName", layer_pair.first);
    PushFrame(layer_pair.first, std::move(layer_frame));
  }
}
