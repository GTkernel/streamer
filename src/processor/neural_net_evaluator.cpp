
#include "neural_net_evaluator.h"

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

StreamPtr NeuralNetEvaluator::GetSink() {
  return Processor::GetSink(SINK_NAME);
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
    if (frame->Count(input_layer_name_) > 0) {
      cur_batch_.push_back(frame->GetValue<cv::Mat>(input_layer_name_));
    } else {
      cur_batch_.push_back(frame->GetValue<cv::Mat>("image"));
    }
  }

  auto start_time = boost::posix_time::microsec_clock::local_time();
  auto layer_outputs = model_->Evaluate({{input_layer_name_, cur_batch_}},
                                        output_layer_names_);
  long time_elapsed =
      (boost::posix_time::microsec_clock::local_time() - start_time)
          .total_microseconds();

  // Push the activations for each published layer to their respective sink.
  for (decltype(cur_batch_frames_.size()) i = 0; i < cur_batch_frames_.size();
       ++i) {
    std::unique_ptr<Frame> ret_frame = std::move(cur_batch_frames_.at(i));
    for (const auto& layer_pair : layer_outputs) {
      auto activation_vector = layer_pair.second;
      auto layer_name = layer_pair.first;
      cv::Mat activations = activation_vector.at(i);
      ret_frame->SetValue(layer_name, activations);
      ret_frame->SetValue("neural_net_evaluator.inference_time_micros",
                          time_elapsed);
    }
    PushFrame(SINK_NAME, std::move(ret_frame));
  }
  cur_batch_frames_.clear();
}
