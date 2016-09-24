//
// Created by Ran Xian on 8/1/16.
//

#ifndef TX1DNN_CAFFE_CLASSIFIER_H
#define TX1DNN_CAFFE_CLASSIFIER_H

#include <caffe/caffe.hpp>
#include "classifier.h"
#include "common/common.h"

/**
 * @brief Vanila Caffe classifier. This classifier is compatible with Caffe V1
 * interfaces. It could be built on both CPU and GPU (unlike CaffeFp16Classifier
 * which can only be built on GPU).
 */
template <typename DType>
class CaffeClassifier : public Classifier {
 public:
  CaffeClassifier(const string &model_file, const string &trained_file,
                  const string &mean_file, const string &label_file);

  virtual size_t GetInputBufferSize() { return GetInputSize<float>(); }

 private:
  void SetMean(const string &mean_file);
  virtual std::vector<float> Predict();
  virtual DataBuffer GetInputBuffer();

 private:
  std::shared_ptr<caffe::Net<DType>> net_;
};

#endif  // TX1DNN_CaffeClassifier_H
