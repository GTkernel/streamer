
#include "neural_net_evaluator.h"

#include "model/model_manager.h"
#include "utils/string_utils.h"

constexpr auto SOURCE_NAME = "input";

NeuralNetEvaluator::NeuralNetEvaluator(
    const ModelDesc& model_desc, const Shape& input_shape, size_t batch_size,
    const std::vector<std::string>& output_layer_names)
    : Processor(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR, {SOURCE_NAME}, {}),
      input_shape_(input_shape),
      batch_size_(batch_size) {
  // Load model.
  auto& manager = ModelManager::GetInstance();
  model_ = manager.CreateModel(model_desc, input_shape_, batch_size_);
  model_->Load();

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
  return std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                              output_layer_names);
}

bool NeuralNetEvaluator::Init() { return true; }

bool NeuralNetEvaluator::OnStop() { return true; }

void NeuralNetEvaluator::SetSource(const std::string& name, StreamPtr stream,
                                   const std::string& layername) {
  if (layername == "") {
    input_layer_name_ = model_->GetModelDesc().GetDefaultInputLayer();
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

void NeuralNetEvaluator::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  cv::Mat input_mat;
  cur_batch_frames_.push_back(std::move(input_frame));
  if (cur_batch_frames_.size() < batch_size_) {
    return;
  }
  std::vector<cv::Mat> cur_batch_;
  for (auto& frame : cur_batch_frames_) {
    if (frame->Count("activations") > 0) {
      cur_batch_.push_back(frame->GetValue<cv::Mat>("activations"));
    } else {
      cur_batch_.push_back(frame->GetValue<cv::Mat>("image"));
    }
  }

  std::vector<std::string> output_layer_names;
  for (const auto& sink_pair : sinks_) {
    output_layer_names.push_back(sink_pair.first);
  }

  auto start_time = boost::posix_time::microsec_clock::local_time();
  auto layer_outputs =
      model_->Evaluate({{input_layer_name_, cur_batch_}}, output_layer_names);
  long time_elapsed =
      (boost::posix_time::microsec_clock::local_time() - start_time)
          .total_microseconds();

  // Push the activations for each published layer to their respective sink.
  for (const auto& layer_pair : layer_outputs) {
    int batch_idx = 0;
    auto activation_vector = layer_pair.second;
    for (const auto& activations : activation_vector) {
      auto layer_name = layer_pair.first;
      std::unique_ptr<Frame> frame_copy = std::make_unique<Frame>(cur_batch_frames_.at(batch_idx++));
      frame_copy->SetValue("activations", activations);
      frame_copy->SetValue("activations_layer_name",
                                             layer_name);
      frame_copy->SetValue(
          "neural_net_evaluator.inference_time_micros", time_elapsed);
      PushFrame(layer_name, std::move(frame_copy));
    }
  }
  cur_batch_frames_.clear();
}
