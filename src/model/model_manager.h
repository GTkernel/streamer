//
// Created by xianran on 9/24/16.
//

#ifndef TX1DNN_MODEL_CONTROLLER_H
#define TX1DNN_MODEL_CONTROLLER_H

#include "model.h"
#include "common/common.h"

#include <unordered_map>

/**
 * @brief A singleton class that controls all models.
 */
class ModelManager {
 public:
  static ModelManager &GetInstance();
 public:
  ModelManager();
  std::vector<int> GetMeanColors() const;
  std::unordered_map<string, ModelDesc> GetModelDescs() const;
  ModelDesc GetModelDesc(const string &name) const;
 private:
  // Mean colors, in BGR order.
  std::vector<int> mean_colors_;
  std::unordered_map<string, ModelDesc> model_descs_;
};

#endif //TX1DNN_MODEL_CONTROLLER_H
