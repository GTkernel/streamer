//
// Created by Ran Xian on 8/1/16.
//

#ifndef TX1DNN_CAFFEV1CLASSIFIER_H
#define TX1DNN_CAFFEV1CLASSIFIER_H

#undef CPU_ONLY
#include <caffe/caffe.hpp>
#include "common.h"
#include "classifier.h"

/**
 * \brief Caffe classifier. 
 */
template <typename DType>
class CaffeV1Classifier : public Classifier {
public:
  CaffeV1Classifier(const string& model_file,
                    const string& trained_file,
                    const string& mean_file,
                    const string& label_file);

private:
  void SetMean(const string& mean_file);
  virtual std::vector<float> Predict();
  virtual DataBuffer GetInputBuffer();

private:
  std::shared_ptr<caffe::Net<DType> > net_;
};

#endif //TX1DNN_CAFFEV1CLASSIFIER_H
