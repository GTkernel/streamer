//
// Created by Ran Xian on 7/27/16.
//

#include <iostream>
#include <fstream>

#include "gie_classifier.h"
#include "utils.h"

GIEClassifier::GIEClassifier(const string &deploy_file,
                             const string &model_file,
                             const string &mean_file,
                             const string &label_file)
    : Classifier(deploy_file, model_file, mean_file, label_file),
      inferer_(deploy_file, model_file, "data", "prob") {
  inferer_.CreateEngine();
  // Set dimensions
  input_channels_ = inferer_.GetInputShape().channel;
  input_width_ = inferer_.GetInputShape().width;
  input_height_ = inferer_.GetInputShape().height;

  // Load the binaryproto mean file.
  SetMean(mean_file);

  // Allocate input data and output data
  input_buffer_ = DataBuffer(GetInputSize<float>());
  output_buffer_ = DataBuffer(inferer_.GetOutputShape().GetVolume() * sizeof(float));
}

GIEClassifier::~GIEClassifier() {
  inferer_.DestroyEngine();
}

/**
 * There is some code duplication, need refactor.
 */
void GIEClassifier::SetMean(const string &mean_file) {
  IBinaryProtoBlob *meanBlob = CaffeParser::parseBinaryProto(mean_file.c_str());
  Dims4 mean_blob_dim = meanBlob->getDimensions();
  const float *data = reinterpret_cast<const float *>(meanBlob->getData());
  float *mutable_data = new float[mean_blob_dim.w * mean_blob_dim.h * mean_blob_dim.c];
  memcpy(mutable_data, data, (size_t)mean_blob_dim.w * mean_blob_dim.h * mean_blob_dim.c);
  float *mutable_data_ptr = mutable_data;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < input_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob_dim.h, mean_blob_dim.w, CV_32FC1, mutable_data_ptr);
    channels.push_back(channel);
    mutable_data_ptr += mean_blob_dim.h * mean_blob_dim.w;
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_image_ = cv::Mat(GetInputGeometry(), mean.type(), channel_mean);

  meanBlob->destroy();
  delete[] mutable_data;
}

std::vector<float> GIEClassifier::Predict() {
  inferer_.DoInference((DType *)input_buffer_.GetBuffer(), (DType *)output_buffer_.GetBuffer());
  int output_channels = inferer_.GetOutputShape().channel;
  std::vector<float> scores;
  for (int i = 0; i < output_channels; i++) {
    scores.push_back(((float *)output_buffer_.GetBuffer())[i]);
  }
  return scores;
}

DataBuffer GIEClassifier::GetInputBuffer() {
  return input_buffer_;
}
