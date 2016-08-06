//
// Created by Ran Xian on 7/23/16.
//

#ifndef TX1DNN_CAFFECLASSIFIER_H
#define TX1DNN_CAFFECLASSIFIER_H

#include <caffe/caffe.hpp>
#include "common.h"
#include "classifier.h"

/**
 * @brief Caffe classifier. DType is either float16 or float. We will temporarily use Caffe's float16, but will want to
 * use NVDIA's own float16 type, or have our own wrapper. MType is either float or CAFFE_FP16_MTYPE.
 */
class CaffeFp16Classifier : public Classifier {
  typedef caffe::float16 DType;
  typedef CAFFE_FP16_MTYPE MType;
 public:
  CaffeFp16Classifier(const string& model_file,
                  const string& trained_file,
                  const string& mean_file,
                  const string& label_file);

 private:
  void SetMean(const string& mean_file);
  virtual std::vector<float> Predict();
  virtual DataBuffer GetInputBuffer();
  virtual void Preprocess(const cv::Mat &img, DataBuffer &buffer);

 private:
  std::shared_ptr<caffe::Net<DType, MType>> net_;
};

#endif //TX1DNN_CAFFECLASSIFIER_H
