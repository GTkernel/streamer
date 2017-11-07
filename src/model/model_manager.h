//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_MODEL_MANAGER_H_
#define STREAMER_MODEL_MODEL_MANAGER_H_

#include <unordered_map>

#include <opencv2/opencv.hpp>

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
  std::unordered_map<std::string, std::vector<ModelDesc>> GetAllModelDescs()
      const;
  ModelDesc GetModelDesc(const std::string& name) const;
  std::vector<ModelDesc> GetModelDescs(const std::string& name) const;
  bool HasModel(const std::string& name) const;
  std::unique_ptr<Model> CreateModel(const ModelDesc& model_desc,
                                     Shape input_shape, size_t batch_size = 1);

 private:
  // Mean colors, in BGR order.
  cv::Scalar mean_colors_;
  std::unordered_map<std::string, std::vector<ModelDesc>> model_descs_;
};

#endif  // STREAMER_MODEL_MODEL_MANAGER_H_
