//
// Created by Ran Xian on 8/5/16.
//

#include "classifier.h"
#include "utils.h"

Classifier::Classifier(const string &label_file) {
  // Load labels.
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
}

/**
 * @brief Classify the image and return top N predictions.
 * @param img The image to be classified.
 * @param N Number of predictions.
 * @return An array of N prediction, sorted by inference score.
 */
std::vector<Prediction> Classifier::Classify(const cv::Mat &img, int N) {
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
 * @brief Transform a image to a given channel, width and height. In the mean while, convert pixel value to float.
 * @param img The image to be transformed.
 * @param channel The channel of the expected output image.
 * @param width The width of the expeceted output image.
 * @param height The height of the expected output image.
 * @return The transformed image.
 */
cv::Mat Classifier::TransformImage(const cv::Mat &img, int channel, int width, int height) {
  cv::Mat sample;
  if (img.channels() == 3 && channel == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && channel == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && channel == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && channel == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  cv::Size input_geometry(width, height);
  if (sample.size() != input_geometry)
    cv::resize(sample, sample_resized, input_geometry);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (channel == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  return sample_float;
}
