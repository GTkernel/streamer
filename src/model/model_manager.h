//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_MODEL_MANAGER_H_
#define STREAMER_MODEL_MODEL_MANAGER_H_

#include "common/common.h"
#include "model.h"

#include <unordered_map>

/**
 * @brief A singleton class that controls all models.
 */
class ModelManager {
 public:
  static ModelManager& GetInstance();

 public:
  ModelManager();
  ModelManager(const ModelManager& other) = delete;
  std::vector<int> GetMeanColors() const;
  //std::unordered_map<string, ModelDesc> GetModelDescs() const;
  ModelDesc GetModelDesc(const string& name) const;
  std::vector<ModelDesc> GetModelDescs(const string& name) const;
  bool HasModel(const string& name) const;
  std::unique_ptr<Model> CreateModel(const ModelDesc& model_desc,
                                     Shape input_shape, size_t batch_size = 1);

 private:
  // Mean colors, in BGR order.
  std::vector<int> mean_colors_;
  std::unordered_map<string, std::vector<ModelDesc>> model_descs_;
};

#endif  // STREAMER_MODEL_CONTROLLER_H
