//
// Created by xianran on 9/24/16.
//

#ifndef TX1DNN_MODEL_DESC_H
#define TX1DNN_MODEL_DESC_H

#include "common/common.h"

/**
 * @brief A decription of a DNN model, created from models.toml file. A ModelDesc
 * can be used to initialize a model.
 */
class ModelDesc {
 public:
  ModelDesc() {}
  ModelDesc(const string &name,
            const ModelType &type,
            const string &model_desc_path,
            const string &model_params_path,
            int input_width,
            int input_height);
  const string &GetName() const;
  const ModelType &GetType() const;
  const string &GetModelDescPath() const;
  const string &GetModelParamsPath() const;
  int GetInputWidth() const;
  int GetInputHeight() const;
 private:
  string name_;
  ModelType type_;
  string model_desc_path_;
  string model_params_path_;
  int input_width_;
  int input_height_;
};

#endif //TX1DNN_MODEL_DESC_H
