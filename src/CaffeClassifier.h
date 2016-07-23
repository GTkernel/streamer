//
// Created by Ran Xian on 7/23/16.
//

#ifndef TX1DNN_CAFFECLASSIFIER_H
#define TX1DNN_CAFFECLASSIFIER_H

// Uncomment below for development
#include <caffe/caffe.hpp>
#include "common.h"

class CaffeClassifier {
public:
    typedef std::pair<string, float> Prediction;
    CaffeClassifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file);
    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    std::shared_ptr<caffe::Net<float, float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};

#endif //TX1DNN_CAFFECLASSIFIER_H
