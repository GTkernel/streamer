//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "mxnet_model.h"
MXNetModel::MXNetModel(const ModelDesc &model_desc, Shape input_shape) : Model(
    model_desc,
    input_shape) {}

void MXNetModel::Load() {
  // Load the model desc and weights
  DataBuffer json_data(model_desc_.GetModelDescPath());
  DataBuffer param_data(model_desc_.GetModelParamsPath());

#ifdef CPU_ONLY
  int dev_type = 1; // CPU
#else
  int dev_type = 2; // GPU
#endif

  int dev_id = 0;
  mx_uint num_input_nodes = 1;
  const char *input_key[1] = {"data"};
  const char **input_keys = input_key;

  const mx_uint input_shape_indptr[2] = {0, 4};
  const mx_uint
      input_shape_data[4] = {1, static_cast<mx_uint>(input_shape_.channel),
                             static_cast<mx_uint>(input_shape_.width),
                             static_cast<mx_uint>(input_shape_.height)};

  int r = MXPredCreate((const char *) json_data.GetBuffer(),
                       (const char *) param_data.GetBuffer(),
                       static_cast<size_t>(param_data.GetSize()),
                       dev_type,
                       dev_id,
                       num_input_nodes,
                       input_keys,
                       input_shape_indptr,
                       input_shape_data,
                       &predictor_);
  if (r < 0) {
    LOG(FATAL) << "Can't initialize MXNet model " << MXGetLastError();
  }

  input_buffer_ = DataBuffer(input_shape_.GetSize() * sizeof(mx_float));
}
void MXNetModel::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  size_t image_size = input_shape_.GetSize();
  MXPredSetInput(predictor_, "data", (mx_float *) input_buffer_.GetBuffer(),
                 image_size);
  MXPredForward(predictor_);

  // Don't know how to get multiple output..
  mx_uint mx_output_index = 0;
  mx_uint *mx_output_shape = 0;
  mx_uint mx_output_shape_len;

  // Get Output Result
  MXPredGetOutputShape(predictor_, mx_output_index, &mx_output_shape,
                       &mx_output_shape_len);

  // THE FOLLOWING ASSUMES THAT OUTPUT SHAPE WILL NEVER EXCEEDS 3-D
  CHECK(mx_output_shape_len <= 3);
  Shape output_shape(1, 1, 1);
  if (mx_output_shape_len > 0)
    output_shape.channel = mx_output_shape[0];
  if (mx_output_shape_len > 1)
    output_shape.height = mx_output_shape[1];
  if (mx_output_shape_len > 2)
    output_shape.width = mx_output_shape[2];

  output_shapes_.push_back(output_shape);

  DataBuffer output_buffer(output_shape.GetSize() * sizeof(float));
  MXPredGetOutput(predictor_,
                  mx_output_index,
                  (mx_float *) output_buffer.GetBuffer(),
                  output_shape.GetSize());
  output_buffers_.push_back(output_buffer);
}

MXNetModel::~MXNetModel() {
  // Release Predictor
  if (predictor_ != nullptr) {
    MXPredFree(predictor_);
    predictor_ = nullptr;
  }
}

