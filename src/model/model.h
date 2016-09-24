//
// Created by xianran on 9/24/16.
//

#ifndef TX1DNN_MODEL_H
#define TX1DNN_MODEL_H

#include "common/common.h"
#include "model_desc.h"

/**
 * @brief A class representing a DNN model. Currently only Caffe model is supported.
 * // TODO: Implement CaffeModel
 */
class Model {
 public:
  Model(ModelDesc model_desc);
  ModelDesc GetModelDesc() const;
  virtual void Init() = 0;
  virtual void Evaluate() = 0;
  virtual void Destroy() = 0;
 private:
  ModelDesc model_desc_;
};

#endif //TX1DNN_MODEL_H
