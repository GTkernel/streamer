//
// Created by Ran Xian on 8/1/16.
//

#ifndef TX1DNN_CAFFEV1CLASSIFIER_H
#define TX1DNN_CAFFEV1CLASSIFIER_H

#undef CPU_ONLY
#include <caffe/caffe.hpp>
#include "common.h"

/**
 * \brief Caffe classifier. 
 */
template <typename DType>
class CaffeV1Classifier {
public:
    CaffeV1Classifier(const string& model_file,
                    const string& trained_file,
                    const string& mean_file,
                    const string& label_file);
    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<DType *> &input_channels);

    void Preprocess(const cv::Mat& img,
                    const std::vector<DType *> &input_channels);

private:
  std::shared_ptr<caffe::Net<DType> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif //TX1DNN_CAFFEV1CLASSIFIER_H
