//
// Created by Ran Xian on 8/4/16.
//

#ifndef TX1DNN_MXNET_CLASSIFIER_H
#define TX1DNN_MXNET_CLASSIFIER_H

#include "common.h"
#include <mxnet/c_predict_api.h>
#include <mxnet/c_api.h>

/**
 * \brief MXNet classifier
 */
class MXNetClassifier {
 public:
  MXNetClassifier(const string &model_desc,
                  const string &model_params,
                  const string &mean_file,
                  const string &label_file,
                  const int input_width,
                  const int input_height);
  ~MXNetClassifier();

  std::vector<Prediction> Classify(const cv::Mat &image, int N = 5);

 private:
  void Preprocessing(const cv::Mat &image, mx_float *input_data);

 private:
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
  PredictorHandle predictor_;
};

#endif //TX1DNN_MXNET_CLASSIFIER_H
