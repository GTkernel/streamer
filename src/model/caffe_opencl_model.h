//
// Created by xianran on 10/1/16.
//

#ifndef TX1DNN_CAFFE_OPENCL_MODEL_H
#define TX1DNN_CAFFE_OPENCL_MODEL_H

#include <caffe/caffe.hpp>
#include "model.h"

template<typename DType>
class CaffeOpenCLModel : public Model {
 public:
  CaffeOpenCLModel(const ModelDesc &model_desc, Shape input_shape);
  virtual void Load();
  virtual void Evaluate();
 private:
  std::unique_ptr<caffe::Net<DType>> net_;
};

#endif //TX1DNN_CAFFE_OPENCL_MODEL_H
