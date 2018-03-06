//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_MODEL_CAFFE_MODEL_H_
#define STREAMER_MODEL_CAFFE_MODEL_H_

#include <unordered_map>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include "model.h"

/**
 * @brief BVLC Caffe model. This model is compatible with Caffe V1
 * interfaces. It could be built on both CPU and GPU.
 */

template <typename DType>
class CaffeModel : public Model {
 public:
  CaffeModel(const ModelDesc& model_desc, Shape input_shape,
             size_t batch_size = 1);
  virtual void Load() override;
  virtual std::unordered_map<std::string, std::vector<cv::Mat>> Evaluate(
      const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map,
      const std::vector<std::string>& output_layer_names) override;

 private:
  std::unique_ptr<caffe::Net<DType>> net_;

  cv::Mat BlobToMat2d(caffe::Blob<DType>* src, int batch_idx) const;
  cv::Mat BlobToMat4d(caffe::Blob<DType>* src, int batch_idx) const;
  cv::Mat GetLayerOutput(const std::string& layer_name, int batch_idx) const;
};

#endif  // STREAMER_MODEL_CAFFE_MODEL_H_
