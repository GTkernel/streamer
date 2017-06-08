//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "mxnet_model.h"
#include "common/context.h"
MXNetModel::MXNetModel(const ModelDesc& model_desc, Shape input_shape,
                       int batch_size)
    : Model(model_desc, input_shape, batch_size) {}

void MXNetModel::Load() {
  // Load the model desc and weights
  DataBuffer json_data(model_desc_.GetModelDescPath());
  DataBuffer param_data(model_desc_.GetModelParamsPath());

  int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);
  int dev_type, dev_id;

  if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
    dev_type = 1;  // CPU
    dev_id = 0;
  } else {
    dev_type = 2;  // GPU
    dev_id = desired_device_number;
  }

  mx_uint num_input_nodes = 1;
  LOG(WARNING) << "Only one input named `data' is supported";
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  const mx_uint input_shape_indptr[2] = {0, 4};
  const mx_uint input_shape_data[4] = {
      batch_size_, static_cast<mx_uint>(input_shape_.channel),
      static_cast<mx_uint>(input_shape_.width),
      static_cast<mx_uint>(input_shape_.height)};

  int r = MXPredCreate((const char*)json_data.GetBuffer(),
                       (const char*)param_data.GetBuffer(),
                       static_cast<size_t>(param_data.GetSize()), dev_type,
                       dev_id, num_input_nodes, input_keys, input_shape_indptr,
                       input_shape_data, &predictor_);
  if (r < 0) {
    LOG(FATAL) << "Can't initialize MXNet model " << MXGetLastError();
  }

  LOG(INFO) << "MXNet model initialized";

  input_buffer_ =
      DataBuffer(input_shape_.GetSize() * sizeof(mx_float) * batch_size_);
}

void MXNetModel::Forward() { Evaluate(); }

void MXNetModel::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  MXPredSetInput(predictor_, "data", (mx_float*)input_buffer_.GetBuffer(),
                 input_buffer_.GetSize() / sizeof(mx_float));
  MXPredForward(predictor_);

  // Don't know how to get multiple output..
  mx_uint mx_output_index = 0;
  mx_uint* mx_output_shape = 0;
  mx_uint mx_output_shape_len;

  // Get Output Result
  MXPredGetOutputShape(predictor_, mx_output_index, &mx_output_shape,
                       &mx_output_shape_len);

  // THE FOLLOWING ASSUMES THAT OUTPUT SHAPE WILL NEVER EXCEEDS 3-D
  CHECK(mx_output_shape_len <= 4 && mx_output_shape_len >= 1);
  Shape output_shape(1, 1, 1);

  int batch_size;

  if (mx_output_shape_len > 0) batch_size = mx_output_shape[0];
  if (mx_output_shape_len > 1) output_shape.channel = mx_output_shape[1];
  if (mx_output_shape_len > 2) output_shape.width = mx_output_shape[2];
  if (mx_output_shape_len > 3) output_shape.height = mx_output_shape[3];

  DLOG(INFO) << "Output shape is " << output_shape.channel << " "
             << output_shape.width << " " << output_shape.height;
  DLOG(INFO) << "Batch size is " << batch_size;
  output_shapes_.push_back(output_shape);

  DataBuffer output_buffer(output_shape.GetSize() * sizeof(mx_float) *
                           batch_size_);
  MXPredGetOutput(predictor_, mx_output_index,
                  (mx_float*)output_buffer.GetBuffer(),
                  output_buffer.GetSize() / sizeof(mx_float));
  output_buffers_.push_back(output_buffer);
}

const std::vector<std::string>& MXNetModel::GetLayerNames() const {
  STREAMER_NOT_IMPLEMENTED;
}

cv::Mat MXNetModel::GetLayerOutput(const std::string&) const {
  STREAMER_NOT_IMPLEMENTED;
}

MXNetModel::~MXNetModel() {
  // Release Predictor
  if (predictor_ != nullptr) {
    MXPredFree(predictor_);
    predictor_ = nullptr;
  }
}
