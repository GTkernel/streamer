//
// Created by Ran Xian on 8/4/16.
//

#include "mxnet_classifier.h"
#include "common/utils.h"

#define MEAN_VALUE 117.0

/**
 * @brief Constructor for MXNet classifier
 * 
 * @param model_desc Model description json file.
 * @param model_params Model param nd file.
 * @param mean_file Mean image nd file.
 * @param label_file Label list file.
 * @param input_width Width of input data.
 * @param input_height Height of input data.
 */
MXNetClassifier::MXNetClassifier(const string &model_desc,
                                 const string &model_params,
                                 const string &mean_file,
                                 const string &label_file,
                                 const int input_width,
                                 const int input_height)
    : Classifier(model_desc, model_params, mean_file, label_file),
      predictor_(nullptr) {

  input_width_ = input_width;
  input_height_ = input_height;
  input_channels_ = 3;
  // Load the model desc and weights
  DataBuffer json_data(model_desc);
  DataBuffer param_data(model_params);

  int dev_type = 2; // GPU
  int dev_id = 0;
  mx_uint num_input_nodes = 1;
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(3),
                                        static_cast<mx_uint>(input_width),
                                        static_cast<mx_uint>(input_height) };

  MXPredCreate((const char*)json_data.GetBuffer(),
               (const char*)param_data.GetBuffer(),
               static_cast<size_t>(param_data.GetSize()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &predictor_);

  mean_image_ = cv::Mat(GetInputGeometry(), CV_32FC3, MEAN_VALUE);

  input_buffer_ = DataBuffer(GetInputSize<mx_float>());
}

MXNetClassifier::~MXNetClassifier() {
  // Release Predictor
  if (predictor_ != nullptr) {
    MXPredFree(predictor_);
    predictor_ = nullptr;
  }
}

std::vector<float> MXNetClassifier::Predict() {
  size_t image_size = GetInputShape().GetVolume();
  MXPredSetInput(predictor_, "data", (mx_float *)input_buffer_.GetBuffer(), image_size);
  MXPredForward(predictor_);

  mx_uint output_index = 0;
  mx_uint *output_shape = 0;
  mx_uint output_shape_len;

  // Get Output Result
  MXPredGetOutputShape(predictor_, output_index, &output_shape, &output_shape_len);

  size_t output_size = 1;
  for (mx_uint i = 0; i < output_shape_len; ++i)
    output_size *= output_shape[i];

  std::vector<float> output(output_size);

  MXPredGetOutput(predictor_, output_index, output.data(), output_size);

  return output;
}

DataBuffer MXNetClassifier::GetInputBuffer() {
  return input_buffer_;
}
