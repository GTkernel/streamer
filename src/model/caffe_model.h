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
  virtual const std::vector<std::string> &GetLayerNames() const override;
  virtual cv::Mat GetLayerOutput(const std::string &layer_name) const override;

 private:
  std::unique_ptr<caffe::Net<DType>> net_;

  static cv::Mat BlobToMat2d(caffe::Blob<DType> *src);
  static cv::Mat BlobToMat4d(caffe::Blob<DType> *src);
};

#endif  // STREAMER_CAFFE_MODEL_H
