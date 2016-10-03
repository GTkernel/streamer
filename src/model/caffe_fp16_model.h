//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef TX1DNN_CAFFE_FP16_MODEL_H
#define TX1DNN_CAFFE_FP16_MODEL_H

#include <caffe/caffe.hpp>
#include "model.h"

/**
 * @brief Caffe half precision classifier. DType is either float16 or float.
 * We will temporarily use Caffe's float16, but will want to use NVDIA's own
 * float16 type, or have our own wrapper. MType is either float or
 * CAFFE_FP16_MTYPE.
 */
class CaffeFp16Model : public Model {
  typedef caffe::float16 DType;
  typedef CAFFE_FP16_MTYPE MType;

 public:
  CaffeFp16Model(const ModelDesc &model_desc, Shape input_shape);
  virtual void Load();
  virtual void Evaluate();

 private:
  std::shared_ptr<caffe::Net<DType, MType>> net_;
};

#endif  // TX1DNN_CAFFE_FP16_MODEL_H
