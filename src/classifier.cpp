//
// Created by Ran Xian on 8/5/16.
//

#include "classifier.h"
#include "utils.h"

Classifier::Classifier(const string &model_desc, const string &model_params, const string &mean_file, const string &label_file) {
  // Load labels.
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
}

std::vector<Prediction> Classifier::Classify(const cv::Mat &img, int N) {
  Timer total_timer;
  Timer timer;

  total_timer.Start();
  timer.Start();
  auto input_buffer = GetInputBuffer();
  Preprocess(img, input_buffer);
  LOG(INFO) << "Preprocess done in " << timer.ElapsedMSec() << " ms";

  timer.Start();
  std::vector<float> output = Predict();
  LOG(INFO) << "Predict done in " << timer.ElapsedMSec() << " ms";

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  LOG(INFO) << "Whole classify done in " << total_timer.ElapsedMSec() << " ms";
  return predictions;
}

std::vector<Prediction> Classifier::Classify(const DataBuffer &buffer, int N) {
  Timer total_timer; total_timer.Start();
  Timer timer;

  total_timer.Start();

  auto input_buffer = GetInputBuffer();
  input_buffer.Clone(buffer);

  timer.Start();
  std::vector<float> output = Predict();
  LOG(INFO) << "Predict done in " << timer.ElapsedMSec() << " ms";

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  LOG(INFO) << "Whole classify done in " << total_timer.ElapsedMSec() << " ms";
  return predictions;
}

/**
 * @brief Transform and normalize an image and flatten the bytes of the image to a data buffer if provided
 * @details The image will first be resized to the given shape, and then substract the mean image, finally flattend to a buffer.
 * 
 * @param img The image to be transformed, can be either in RGB, RGBA, or gray scale
 * @param shape The wanted shape of the transformed image.
 * @param mean_img Mean image. If don't want to substract mean image, provide a *zero* image, (e.g. cv::Mat::zeros(channel, height, width)).
 * @param buffer The buffer to store the transformed image. No storage will happen if \b buffer is nullptr.
 * @return The transformed image.
 */
cv::Mat Classifier::TransformImage(const cv::Mat &img, const Shape &shape, const cv::Mat &mean_img, DataBuffer *buffer) {
  CHECK(mean_img.channels() == shape.channel && mean_img.size[0] == shape.width && mean_img.size[1] == shape.height)
    << "Mean image shape does not match that of desired shape";
  int num_channel = shape.channel, width = shape.width, height = shape.height;

  // Convert channels
  cv::Mat sample;
  if (img.channels() == 3 && num_channel == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channel == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channel == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channel == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  // Resize
  cv::Mat sample_resized;
  cv::Size input_geometry(width, height);
  if (sample.size() != input_geometry)
    cv::resize(sample, sample_resized, input_geometry);
  else
    sample_resized = sample;

  // Convert to float
  cv::Mat sample_float;
  if (num_channel == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  // Normalize
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_img, sample_normalized);

  if (buffer != nullptr) {
    LOG(INFO) << "Copy transformed input image to input buffer";
    CHECK(shape.GetVolume() * sizeof(float) == buffer->GetSize()) << "Buffer size " << buffer->GetSize()
      << " does not match input size " << shape.GetVolume() * sizeof(float);
    // Wrap buffer to channels to save memory copy.
    float *data = (float *)buffer->GetBuffer();
    std::vector<cv::Mat> output_channels;
    for (int i = 0; i < num_channel; i++) {
      cv::Mat channel(height, width, CV_32FC1, data);
      output_channels.push_back(channel);
      data += width * height;
    }
    cv::split(sample_normalized, output_channels);
  }

  return sample_float;
}

void Classifier::Preprocess(const cv::Mat &img, DataBuffer &buffer) {
  TransformImage(img, GetInputShape(), mean_image_, &buffer);
}
