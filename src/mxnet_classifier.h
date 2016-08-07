//
// Created by Ran Xian on 8/4/16.
//

#ifndef TX1DNN_MXNET_CLASSIFIER_H
#define TX1DNN_MXNET_CLASSIFIER_H

#include "common.h"
#include "classifier.h"
#include <mxnet/c_predict_api.h>
#include <mxnet/c_api.h>

/**
 * \brief MXNet classifier
 */
class MXNetClassifier : public Classifier {
 public:
  MXNetClassifier(const string &model_desc,
                  const string &model_params,
                  const string &mean_file,
                  const string &label_file,
                  const int input_width,
                  const int input_height);
  ~MXNetClassifier();

  virtual size_t GetInputBufferSize() {
    return GetInputSize<mx_float>();
  }

 private:
  virtual std::vector<float> Predict();
  virtual DataBuffer GetInputBuffer();

 private:
  PredictorHandle predictor_;
  DataBuffer input_buffer_;
};

#endif //TX1DNN_MXNET_CLASSIFIER_H
