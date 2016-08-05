//
// Created by Ran Xian on 7/23/16.
//

#ifndef TX1DNN_CAFFECLASSIFIER_H
#define TX1DNN_CAFFECLASSIFIER_H

#undef CPU_ONLY
#include <caffe/caffe.hpp>
#include "common.h"

#ifdef ON_MAC
#define float16 float
#define CAFFE_FP16_MTYPE float
#else
using caffe::float16;
#endif

/**
 * \brief Caffe classifier. DType is either float16 or float. We will temporarily use Caffe's float16, but will want to
 * use NVDIA's own float16 type, or have our own wrapper. MType is either float or CAFFE_FP16_MTYPE.
 */
template <typename DType, typename MType>
class CaffeClassifier {
public:
    CaffeClassifier(const string& model_file,
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
  std::shared_ptr<caffe::Net<DType, MType> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif //TX1DNN_CAFFECLASSIFIER_H
