
#include "model/tf_model.h"

#include <fstream>

#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include "common/context.h"
#include "utils/utils.h"

TFModel::TFModel(const ModelDesc& model_desc, Shape input_shape)
    : Model(model_desc, input_shape),
      input_op_(model_desc.GetDefaultInputLayer()),
      last_op_(model_desc.GetDefaultOutputLayer()) {}

TFModel::~TFModel() {
  tensorflow::Session* raw = session_.release();
  delete raw;
}

void TFModel::Load() {
  int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);
  if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
    LOG(INFO) << "Use device: " << DEVICE_NUMBER_CPU_ONLY << " (CPU)";
  } else {
    LOG(FATAL) << "Compiled in CPU-only mode but using a device number "
               << "other than -1.";
  }

  // Load the network.
  tensorflow::GraphDef graph_def;
  tensorflow::Status status = ReadBinaryProto(
      tensorflow::Env::Default(), model_desc_.GetModelDescPath(), &graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load TensorFlow graph: " << status.error_message();
  }

  session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  status = session_->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to create TensorFlow Session: "
               << status.error_message();
  }
}

std::unordered_map<std::string, std::vector<tensorflow::Tensor>> TFModel::TensorEvaluate(
    const std::vector<std::pair<std::string, tensorflow::Tensor>> inputs,
    const std::vector<std::string>& output_layer_names) {

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status =
      session_->Run(inputs, {output_layer_names}, {}, &outputs);

  std::unordered_map<std::string, std::vector<tensorflow::Tensor>> ret;
  int count = 0;
  for (auto output_tensor : outputs){
    ret[output_layer_names[count++]].push_back(output_tensor);
  }

  return ret;
}

//OpenCV to tensor, used at the beginning image transforming, would continue with evaluating process
std::vector<std::pair<std::string, tensorflow::Tensor>> TFModel::CV2Tensor(
    const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map) {

  CHECK_EQ(input_map.size(), 1)
      << "Specifying multiple input layers is not supported.";

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  
  // in this case, only one input mapping
  for (const auto& input_pair : input_map) {
    std::string input_layer_name = input_pair.first;
    std::vector<cv::Mat> input_vec = input_pair.second;

    // This is a patch to make tensorflow classification work properly with the
    // current model.
    for (decltype(input_vec.size()) i = 0; i < input_vec.size(); ++i) {
      cv::Mat input = input_vec.at(i);
      cv::Mat input_normalized;
      cv::normalize(input, input_normalized, -0.5, 0.5, cv::NORM_MINMAX);
      input_vec.at(i) = input_normalized;
    }

    cv::Mat input = input_vec.at(0);
    int channel = input.channels();
    int height = input.size[0];
    int width = input.size[1];
    if (input.dims == 4) {
      channel = input.size[3];
      height = input.size[1];
      width = input.size[2];
    }
    // copy data from split (BGR) channels to (RGB) tensor. Datatype must be
    // float. Float16 is not supported yet. Batch size is always 1
    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({static_cast<long long>(input_vec.size()),
                                 height, width, channel}));
    // TODO: Add handling for non-continuous cv mat
    CHECK(input.isContinuous()) << "cv::Mat must be continuous.";
    // This works because the cv::Mat is stored in HWC format. If we want to
    // support CHW format, then we will need to transpose the tensor. It is not
    // clear whether C++ API exports tf.transpose(). Perhaps this will need to
    // be done using Eigen.
    for (decltype(input_vec.size()) i = 0; i < input_vec.size(); i++) {
      std::copy_n(
          (float*)input_vec[i].data, channel * height * width,
          input_tensor.flat<float>().data() + i * channel * height * width);
    }
    // If the input layer is not specified, use the default
    if (input_layer_name == "") {
      input_layer_name = input_op_;
    }
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>(
        input_layer_name, input_tensor));
  }

  return inputs;
}
// the output format transformation for final layer
std::unordered_map<std::string, std::vector<cv::Mat>> TFModel::Tensor2CV(
    const std::unordered_map<std::string, std::vector<tensorflow::Tensor>>& input_map) {

  std::unordered_map<std::string, std::vector<cv::Mat>> ret;  
  
  //only handle single iteration currently
  for (const auto& input_pair : input_map) {
    std::string input_layer_name = input_pair.first;
    std::vector<tensorflow::Tensor> input_vec = input_pair.second;
 
    std::vector<cv::Mat> return_vector;
    for (const auto& output_tensor : input_vec) {
        tensorflow::TensorShape tensor_shape = output_tensor.shape();
        auto batch_size = (*tensor_shape.begin()).size;
        std::vector<int> mat_size;
        size_t vishash_size = 1;
        for (auto it = tensor_shape.begin(); it != tensor_shape.end(); ++it) {
          mat_size.push_back((*it).size);
          vishash_size *= (*it).size;
        }
        for (decltype(batch_size) i = 0; i < batch_size; ++i) {
          cv::Mat temp(mat_size, CV_32F);
          std::copy_n(output_tensor.flat<float>().data() + (i * vishash_size),
                      vishash_size, (float*)temp.data);
  		return_vector.push_back(temp);
        }
  
        ret[input_layer_name] = return_vector;
     }
  }
 
  return ret;
}

std::unordered_map<std::string, std::vector<cv::Mat>> TFModel::Evaluate(
    const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map,
    const std::vector<std::string>& output_layer_names) {
  CHECK_EQ(input_map.size(), 1)
      << "Specifying multiple input layers is not supported.";

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  std::unordered_map<std::string, std::vector<cv::Mat>> ret;

  for (const auto& input_pair : input_map) {
    std::string input_layer_name = input_pair.first;
    std::vector<cv::Mat> input_vec = input_pair.second;

    // This is a patch to make tensorflow classification work properly with the
    // current model.
    for (decltype(input_vec.size()) i = 0; i < input_vec.size(); ++i) {
      cv::Mat input = input_vec.at(i);
      cv::Mat input_normalized;
      cv::normalize(input, input_normalized, -0.5, 0.5, cv::NORM_MINMAX);
      input_vec.at(i) = input_normalized;
    }

    cv::Mat input = input_vec.at(0);
    int channel = input.channels();
    int height = input.size[0];
    int width = input.size[1];
    if (input.dims == 4) {
      channel = input.size[3];
      height = input.size[1];
      width = input.size[2];
    }
    // copy data from split (BGR) channels to (RGB) tensor. Datatype must be
    // float. Float16 is not supported yet. Batch size is always 1
    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({static_cast<long long>(input_vec.size()),
                                 height, width, channel}));
    // TODO: Add handling for non-continuous cv mat
    CHECK(input.isContinuous()) << "cv::Mat must be continuous.";
    // This works because the cv::Mat is stored in HWC format. If we want to
    // support CHW format, then we will need to transpose the tensor. It is not
    // clear whether C++ API exports tf.transpose(). Perhaps this will need to
    // be done using Eigen.
    for (decltype(input_vec.size()) i = 0; i < input_vec.size(); i++) {
      std::copy_n(
          (float*)input_vec[i].data, channel * height * width,
          input_tensor.flat<float>().data() + i * channel * height * width);
    }
    // If the input layer is not specified, use the default
    if (input_layer_name == "") {
      input_layer_name = model_desc_.GetDefaultInputLayer();
    }
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>(
        input_layer_name, input_tensor));
  }

  // Run inference.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status =
      session_->Run(inputs, {output_layer_names}, {}, &outputs);

  if (!status.ok()) {
    LOG(FATAL) << "Session::Run() completed with errors: "
               << status.error_message();
  }

  int count = 0;
  std::vector<cv::Mat> return_vector;
  for (const auto& output_tensor : outputs) {
    tensorflow::TensorShape tensor_shape = output_tensor.shape();
    auto batch_size = (*tensor_shape.begin()).size;
    std::vector<int> mat_size;
    size_t vishash_size = 1;
    for (auto it = tensor_shape.begin(); it != tensor_shape.end(); ++it) {
      mat_size.push_back((*it).size);
      vishash_size *= (*it).size;
    }
    for (decltype(batch_size) i = 0; i < batch_size; ++i) {
      cv::Mat temp(mat_size, CV_32F);
      std::copy_n(output_tensor.flat<float>().data() + (i * vishash_size),
                  vishash_size, (float*)temp.data);
      return_vector.push_back(temp);
    }

    ret[output_layer_names[count++]] = return_vector;
  }
  return ret;
}
