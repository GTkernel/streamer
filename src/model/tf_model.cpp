
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

std::unordered_map<std::string, cv::Mat> TFModel::Evaluate(
    const std::unordered_map<std::string, cv::Mat>& input_map,
    const std::vector<std::string>& output_layer_names) {
  CHECK_EQ(input_map.size(), 1)
      << "Specifying multiple input layers is not supported.";

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  for (const auto& input_pair : input_map) {
    std::string input_layer_name = input_pair.first;
    cv::Mat input = input_pair.second;

    // This is a patch to make tensorflow classification work properly with the
    // current model.
    cv::Mat input_normalized;
    cv::normalize(input, input_normalized, -0.5, 0.5, cv::NORM_MINMAX);

    int channel = input_normalized.channels();
    int height = input_normalized.size[0];
    int width = input_normalized.size[1];
    if (input_normalized.dims == 4) {
      channel = input_normalized.size[3];
      height = input_normalized.size[1];
      width = input_normalized.size[2];
    }
    // copy data from split (BGR) channels to (RGB) tensor. Datatype must be
    // float. Float16 is not supported yet. Batch size is always 1
    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape(
            {static_cast<long long>(1), height, width, channel}));
    auto flat = input_tensor.flat<float>();
    CHECK_EQ(flat.size(), channel * height * width);

    // TODO: add handling for this
    CHECK(input_normalized.isContinuous()) << "cv::Mat must be continuous.";

    // This works because the cv::Mat is stored in HWC format. If we want to
    // support CHW format, then we will need to transpose the tensor. It is not
    // clear whether C++ API exports tf.transpose(). Perhaps this will need to
    // be done using Eigen.
    std::copy_n((float*)input_normalized.data,
                input_tensor.flat<float>().size(),
                input_tensor.flat<float>().data());
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

  std::unordered_map<std::string, cv::Mat> results;
  int count = 0;
  for (const auto& output_tensor : outputs) {
    tensorflow::TensorShape tensor_shape = output_tensor.shape();
    std::vector<int> mat_size;
    for (auto it = tensor_shape.begin(); it != tensor_shape.end(); ++it) {
      mat_size.push_back((*it).size);
    }
    cv::Mat result(mat_size, CV_32F);
    std::copy_n(output_tensor.flat<float>().data(),
                output_tensor.flat<float>().size(), (float*)result.data);

    results[output_layer_names[count++]] = result;
  }
  return results;
}
