//
// Created by Ran Xian on 7/23/16.
//

#ifndef TX1DNN_CAFFEFP16CLASSIFIER_H
#define TX1DNN_CAFFEFP16CLASSIFIER_H

#undef CPU_ONLY
#include <caffe/caffe.hpp>
#include "common.h"

#ifdef ON_MAC
#define float16 float
#define CAFFE_FP16_MTYPE float
#else
using caffe::float16;
#endif

class CaffeFp16Classifier {
public:
  typedef std::pair<string, float> Prediction;
  CaffeFp16Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
  cv::Size GetInputGeometry();

private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<float16 *>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<float16 *>* input_channels);

private:
  std::shared_ptr<caffe::Net<float16, CAFFE_FP16_MTYPE> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif //TX1DNN_CAFFEFP16CLASSIFIER_H
