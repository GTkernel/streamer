//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_MODEL_MANAGER_H_
#define STREAMER_MODEL_MODEL_MANAGER_H_

#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "common/common.h"
#include "model.h"

/**
 * @brief A singleton class that controls all models.
 */
class ModelManager {
 public:
  static ModelManager& GetInstance();

 public:
  ModelManager();
  ModelManager(const ModelManager& other) = delete;
  cv::Scalar GetMeanColors() const;
  void SetMeanColors(cv::Scalar mean_colors);
  std::unordered_map<string, ModelDesc> GetModelDescs() const;
  ModelDesc GetModelDesc(const string& name) const;
  bool HasModel(const string& name) const;
  std::unique_ptr<Model> CreateModel(const ModelDesc& model_desc,
                                     Shape input_shape, size_t batch_size = 1);

 private:
  // Mean colors, in BGR order.
  cv::Scalar mean_colors_;
  std::unordered_map<string, ModelDesc> model_descs_;
};

#endif  // STREAMER_MODEL_MODEL_MANAGER_H_
