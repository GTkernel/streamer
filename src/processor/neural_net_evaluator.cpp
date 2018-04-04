
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
  if (model_desc.GetModelType() == MODEL_TYPE_TENSORFLOW){
    tf_model_ = std::make_unique<TFModel>(model_desc, input_shape_);
    tf_model_->Load();
  }else{
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
    if (tf_model_ != NULL){ // using a tensorflow model
    	input_layer_name_ = tf_model_->GetModelDesc().GetDefaultInputLayer();
    }else{
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

template <typename T> void NeuralNetEvaluator::PassFrame(std::unordered_map<std::string, std::vector<T>> outputs,
                                                         long time_elapsed) {
	// Push the activations for each published layer to their respective sink.
    for (const auto& layer_pair : outputs) {
      int batch_idx = 0;
      auto activation_vector = layer_pair.second;
      auto layer_name = layer_pair.first;
      for (const auto& activations : activation_vector) {
        std::unique_ptr<Frame> frame_copy;
        if (outputs.size() == 1) {
          frame_copy = std::move(cur_batch_frames_.at(batch_idx++));
        } else {
          frame_copy = std::make_unique<Frame>(cur_batch_frames_.at(batch_idx++));
        }
        frame_copy->SetValue("activations", activations);
        frame_copy->SetValue("activations_layer_name", layer_name);
        frame_copy->SetValue("neural_net_evaluator.inference_time_micros",
                             time_elapsed);
        PushFrame(layer_name, std::move(frame_copy));
      }
    }
}


void NeuralNetEvaluator::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  cv::Mat input_mat;
  cur_batch_frames_.push_back(std::move(input_frame));
  if (cur_batch_frames_.size() < batch_size_) {
    return;
  }
  std::vector<cv::Mat> cv_batch_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_batch_;
  auto is_last_layer_ = false;

  for (auto& frame : cur_batch_frames_) {
	//have been evaluated, so stored as activations
	//Maybe storing tensor one in another name
    if (frame->Count("activations") > 0) {
      if (tf_model_ != NULL) {
        tensor_batch_.push_back(std::pair<std::string, tensorflow::Tensor>(
                                input_layer_name_,
                                frame->GetValue<tensorflow::Tensor>("activations")
        ));
      } else {
        cv_batch_.push_back(frame->GetValue<cv::Mat>("activations"));
      }
    } else {
	  //never start evaluating, just before going into a model
      cv_batch_.push_back(frame->GetValue<cv::Mat>("image"));
    }
  }

  std::vector<std::string> output_layer_names;
  for (const auto& sink_pair : sinks_) {
    output_layer_names.push_back(sink_pair.first);
    if (tf_model_ != NULL) {
      if (sink_pair.first == tf_model_->GetModelDesc().GetDefaultOutputLayer()){
        is_last_layer_ = true;
      }
    }
  }
  if (tf_model_ == NULL) {
    auto start_time = boost::posix_time::microsec_clock::local_time();
    auto layer_outputs = model_->Evaluate({{input_layer_name_, cv_batch_}}, output_layer_names);
    long time_elapsed =
        (boost::posix_time::microsec_clock::local_time() - start_time)
            .total_microseconds();
  
    // Push the activations for each published layer to their respective sink.
    PassFrame(layer_outputs, time_elapsed);

  } else { // TF model

    std::unordered_map<std::string, std::vector<tensorflow::Tensor>> layer_outputs;
    auto start_time = boost::posix_time::microsec_clock::local_time();
	
    if (tensor_batch_.size() > 0) {
      layer_outputs = tf_model_->TensorEvaluate(tensor_batch_, output_layer_names);
    } else {
       // get OpenCV input  
      auto tensor_vec_ = tf_model_->CV2Tensor({{input_layer_name_, cv_batch_}});
      layer_outputs = tf_model_->TensorEvaluate(tensor_vec_, output_layer_names);      
    }
    long time_elapsed =
        (boost::posix_time::microsec_clock::local_time() - start_time)
            .total_microseconds();

    if(is_last_layer_){
      auto cv_outputs = tf_model_->Tensor2CV(layer_outputs);
      PassFrame(cv_outputs, time_elapsed);
    } else {
      PassFrame(layer_outputs, time_elapsed);
    }
  }

  cur_batch_frames_.clear();
}
