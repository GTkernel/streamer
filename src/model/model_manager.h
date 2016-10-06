//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef TX1DNN_MODEL_CONTROLLER_H
#define TX1DNN_MODEL_CONTROLLER_H

#include "common/common.h"
#include "model.h"

#include <unordered_map>

/**
 * @brief A singleton class that controls all models.
 */
class ModelManager {
 public:
  static ModelManager &GetInstance();

 public:
  ModelManager();
  ModelManager(const ModelManager &other) = delete;
  std::vector<int> GetMeanColors() const;
  std::unordered_map<string, ModelDesc> GetModelDescs() const;
  ModelDesc GetModelDesc(const string &name) const;
  bool HasModel(const string &name) const;
  std::unique_ptr<Model> CreateModel(const ModelDesc &model_desc,
                                     Shape input_shape, int batch_size = 1);

 private:
  // Mean colors, in BGR order.
  std::vector<int> mean_colors_;
  std::unordered_map<string, ModelDesc> model_descs_;
};

#endif  // TX1DNN_MODEL_CONTROLLER_H
