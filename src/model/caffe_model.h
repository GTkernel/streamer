//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_CAFFE_MODEL_H
#define STREAMER_CAFFE_MODEL_H

#include <caffe/caffe.hpp>
#include "model.h"

/**
 * @brief BVLC Caffe model. This model is compatible with Caffe V1
 * interfaces. It could be built on both CPU and GPU (unlike CaffeFp16Classifier
 * which can only be built on GPU).
 */
template <typename DType>
class CaffeModel : public Model {
 public:
  CaffeModel(const ModelDesc &model_desc, Shape input_shape, int batch_size);
  virtual void Load();
  virtual void Evaluate();
  virtual void Forward();

 private:
  std::unique_ptr<caffe::Net<DType>> net_;
};

#endif  // STREAMER_CAFFE_MODEL_H
