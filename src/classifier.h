//
// Created by Ran Xian on 8/5/16.
//

#ifndef TX1DNN_CLASSIFIER_H
#define TX1DNN_CLASSIFIER_H

#include "common.h"

class Classifier {
 public:
  Classifier(const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  virtual std::vector<float> Predict(const cv::Mat& img) = 0;

 protected:
  std::vector<string> labels_;
  static cv::Mat TransformImage(const cv::Mat &img, int channel, int width, int height);
};

#endif //TX1DNN_CLASSIFIER_H
