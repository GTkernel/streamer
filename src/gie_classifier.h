//
// Created by Ran Xian on 7/27/16.
//

#ifndef TX1DNN_GIECLASSIFIER_H
#define TX1DNN_GIECLASSIFIER_H

#include "common.h"
#include "classifier.h"
#include "gie_inferer.h"
#include "float16.h"

class GIEClassifier : public Classifier {
  typedef float DType;
 public:
  typedef std::pair<string, float> Prediction;
  GIEClassifier(const string &model_file,
                const string &trained_file,
                const string &mean_file,
                const string &label_file);
  ~GIEClassifier();

 private:
  void SetMean(const string &mean_file);

  void Preprocess(const cv::Mat &img);

  virtual std::vector<float> Predict(const cv::Mat &img);

 private:
  GIEInferer<DType> inferer_;
  cv::Size input_geometry_;
  size_t num_channels_;
  cv::Mat mean_;
  DType *input_data_;
  DType *output_data_;
};

#endif //TX1DNN_GIECLASSIFIER_H
