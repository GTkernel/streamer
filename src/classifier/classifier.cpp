//
// Created by Ran Xian on 8/5/16.
//

#include "classifier.h"
#include "common/utils.h"
#include "model/model_manager.h"

Classifier::Classifier(std::shared_ptr<Stream> input_stream,
                       const ModelDesc &model_desc,
                       Shape input_shape)
    : input_shape_(input_shape), input_stream_(input_stream), stopped_(false) {
  // Load labels.
  CHECK(model_desc.GetLabelFilePath() != "")
  << "Model " << model_desc.GetName() << " has an empty label file";
  std::ifstream labels(model_desc.GetLabelFilePath());
  CHECK(labels)
  << "Unable to open labels file " << model_desc.GetLabelFilePath();
  string line;
  while (std::getline(labels, line)) labels_.push_back(string(line));

  // Load model
  auto &manager = ModelManager::GetInstance();
  model_ = manager.CreateModel(model_desc, input_shape);
  model_->Load();

  // Create mean image
  auto mean_colors = manager.GetMeanColors();
  mean_image_ =
      cv::Mat(cv::Size(input_shape.width, input_shape.height),
              CV_32FC3,
              cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));

  LOG(INFO) << "Classifier initialized";
}

void Classifier::Start() {
  DataBuffer buffer(input_shape_.GetSize() * sizeof(float));
  while (!stopped_) {
    cv::Mat frame = input_stream_->PopFrame();
    TransformImage(frame, input_shape_, mean_image_, &buffer);
    auto predictions = Classify(buffer, 1);
    for (auto prediction : predictions) {
      LOG(INFO) << prediction.first << " " << prediction.second;
    }
  }
}

void Classifier::Stop() {
  stopped_ = true;
}

std::vector<Prediction> Classifier::Classify(const DataBuffer &buffer, int N) {
  Timer total_timer;
  Timer timer;

  total_timer.Start();
  timer.Start();

  auto input_buffer = model_->GetInputBuffer();
  input_buffer.Clone(buffer);

  model_->Evaluate();
  CHECK(model_->GetOutputBuffers().size() == 1
            && model_->GetOutputShapes().size()
                == model_->GetOutputBuffers().size())
  << "Classify model does not have one buffer";
  float *scores = (float *) model_->GetOutputBuffers()[0].GetBuffer();
  LOG(INFO) << "Predict done in " << timer.ElapsedMSec() << " ms";

  N = std::min<int>(labels_.size(), N);
  std::vector<int>
      maxN = Argmax(scores, model_->GetOutputShapes()[0].GetSize(), N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], scores[idx]));
  }

  LOG(INFO) << "Whole classify done in " << total_timer.ElapsedMSec() << " ms";
  return predictions;
}

/**
 * @brief Transform and normalize an image and flatten the bytes of the image to
 * a data buffer if provided
 * @details The image will first be resized to the given shape, and then
 * substract the mean image, finally flattend to a buffer.
 *
 * @param img The image to be transformed, can be either in RGB, RGBA, or gray
 * scale
 * @param shape The wanted shape of the transformed image.
 * @param mean_img Mean image. If don't want to substract mean image, provide a
 * *zero* image, (e.g. cv::Mat::zeros(channel, height, width)).
 * @param buffer The buffer to store the transformed image. No storage will
 * happen if \b buffer is nullptr.
 * @return The transformed image.
 */
cv::Mat Classifier::TransformImage(const cv::Mat &img, const Shape &shape,
                                   const cv::Mat &mean_img,
                                   DataBuffer *buffer) {
  CHECK(mean_img.channels() == shape.channel &&
      mean_img.size[0] == shape.width && mean_img.size[1] == shape.height)
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

  // Crop according to scale
  int desired_width = (int) ((float) shape.width / shape.height * img.size[1]);
  int desired_height = (int) ((float) shape.height / shape.width * img.size[0]);
  int new_width = img.size[0], new_height = img.size[1];
  if (desired_width < img.size[0]) {
    new_width = desired_width;
  } else {
    new_height = desired_height;
  }
  cv::Rect roi((img.size[1] - new_height) / 2, (img.size[0] - new_width) / 2,
               new_width, new_height);
  cv::Mat sample_cropped = sample(roi);

  // Resize
  cv::Mat sample_resized;
  cv::Size input_geometry(width, height);
  if (sample_cropped.size() != input_geometry)
    cv::resize(sample_cropped, sample_resized, input_geometry);
  else
    sample_resized = sample_cropped;

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
    CHECK(shape.GetSize() * sizeof(float) == buffer->GetSize())
    << "Buffer size " << buffer->GetSize() << " does not match input size "
    << shape.GetSize() * sizeof(float);
    // Wrap buffer to channels to save memory copy.
    float *data = (float *) buffer->GetBuffer();
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
  Timer timer;
  timer.Start();
  TransformImage(img, input_shape_, mean_image_, &buffer);
  LOG(INFO) << "Preprocess takes " << timer.ElapsedMSec() << " ms";
}
