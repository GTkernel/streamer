//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_CAFFE_FP16_MODEL_H
#define STREAMER_CAFFE_FP16_MODEL_H

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
  CaffeFp16Model(const ModelDesc& model_desc, Shape input_shape,
                 int batch_size);
  virtual void Load();
  virtual void Evaluate();
  virtual void Forward();
  virtual const std::vector<std::string>& GetLayerNames() const override;
  virtual cv::Mat GetLayerOutput(const std::string& layer_name) const override;

 private:
  std::shared_ptr<caffe::Net<DType, MType>> net_;
  // Need to transform the input buffer or not
  bool need_transform_;
  // Network input buffer wrapper
  DataBuffer network_input_buffer_;
};

#endif  // STREAMER_CAFFE_FP16_MODEL_H
