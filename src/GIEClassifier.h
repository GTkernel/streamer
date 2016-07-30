//
// Created by Ran Xian on 7/27/16.
//

#ifndef TX1DNN_GIECLASSIFIER_H
#define TX1DNN_GIECLASSIFIER_H

#include "common.h"
#include "GIEClassifier.h"
#include "GIEInferer.h"

class GIEClassifier {
 public:
  typedef std::pair<string, float> Prediction;
  GIEClassifier(const string &model_file,
                const string &trained_file,
                const string &mean_file,
                const string &label_file);
  ~GIEClassifier();
  std::vector<Prediction> Classify(const cv::Mat &img, int N = 5);

 private:
  void SetMean(const string &mean_file);

  std::vector<float> Predict(const cv::Mat &img);

  void CreateInput(const cv::Mat &img);

 private:
  GIEInferer<float> inferer_;
  cv::Size input_geometry_;
  size_t num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  float *input_data_;
  float *output_data_;
};

#endif //TX1DNN_GIECLASSIFIER_H
