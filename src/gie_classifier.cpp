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
                             const string &label_file): inferer_(deploy_file, model_file, "data", "prob") {
  inferer_.CreateEngine();
  // Set dimensions
  input_geometry_ = cv::Size(inferer_.GetInputShape().width, inferer_.GetInputShape().height);
  num_channels_ = inferer_.GetInputShape().channel;
  // Load the binaryproto mean file.
  SetMean(mean_file);

  // Load labels
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  // Allocate input data and output data
  input_data_ = new DType[inferer_.GetInputShape().GetVolume()];
  output_data_ = new DType[inferer_.GetOutputShape().GetVolume()];
}

GIEClassifier::~GIEClassifier() {
  delete[] input_data_;
  delete[] output_data_;
  inferer_.DestroyEngine();
}

std::vector<GIEClassifier::Prediction> GIEClassifier::Classify(const cv::Mat &img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
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
  for (int i = 0; i < num_channels_; ++i) {
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
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);

  meanBlob->destroy();
  delete[] mutable_data;
}

std::vector<float> GIEClassifier::Predict(const cv::Mat &img) {
  Timer micro_timer, macro_timer;
  macro_timer.Start();
  micro_timer.Start();
  CreateInput(img);
  LOG(INFO) << "CreateInput took " << micro_timer.ElapsedMSec() << " ms";
  micro_timer.Start();
  inferer_.DoInference(input_data_, output_data_);
  LOG(INFO) << "DoInference took " << micro_timer.ElapsedMSec() << " ms";
  int output_channels = inferer_.GetOutputShape().channel;
  std::vector<float> scores;
  micro_timer.Start();
  for (int i = 0; i < output_channels; i++) {
    scores.push_back(float(output_data_[i]));
  }
  LOG(INFO) << "Copy output took " << micro_timer.ElapsedMSec() << " ms";
  LOG(INFO) << "Whole prediction done in " << macro_timer.ElapsedMSec() << " ms";
  return scores;
}

void GIEClassifier::CreateInput(const cv::Mat &img) {
  Shape input_shape = inferer_.GetInputShape();
  int channel = input_shape.channel;
  int width = input_shape.width;
  int height = input_shape.height;
  CHECK_EQ(img.channels(), channel) << "Input image channel must equal to network channel";
  CHECK(channel == 3) << "Can only deal with 3-channel BGR images now";
  cv::Mat resized_sample;
  if (img.size[0] != width || img.size[1] == height) {
    cv::resize(img, resized_sample, cv::Size(width, height));
  } else {
    resized_sample = img;
  }

  cv::Mat sample_float;
  resized_sample.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  std::vector<cv::Mat> split_channels(num_channels_);
  cv::split(sample_normalized, split_channels);

  CHECK(split_channels.size() == num_channels_);

  DType *input_pointer = input_data_;
  for (int i = 0; i < num_channels_; i++) {
    for (int j = 0; j < width * height; j++) {
      float input_pixel = ((float *)split_channels[i].data)[j];
      input_pointer[j] = DType(input_pixel);
    }
    input_pointer += width * height;
  }
}