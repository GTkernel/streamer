//
// Created by xianran on 9/29/16.
//

#ifndef TX1DNN_MXNET_MODEL_H
#define TX1DNN_MXNET_MODEL_H

#include "model.h"
#include <mxnet/c_api.h>
#include <mxnet/c_predict_api.h>

/**
 * @brief MXNet model
 */
class MXNetModel : public Model {
 public:
  MXNetModel(const ModelDesc &model_desc, Shape input_shape);
  ~MXNetModel();
  virtual void Load();
  virtual void Evaluate();
 private:
  PredictorHandle predictor_;
};

#endif //TX1DNN_MXNET_MODEL_H