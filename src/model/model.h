//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_MODEL_H_
#define STREAMER_MODEL_MODEL_H_

#include <unordered_map>

#include "common/common.h"
#include "model.h"

/**
 * @brief A decription of a DNN model, created from models.toml file. A
 * ModelDesc can be used to initialize a model.
 */
class ModelDesc {
 public:
  ModelDesc() {}
  ModelDesc(const string& name, const ModelType& type,
            const string& model_desc_path, const string& model_params_path,
            int input_width, int input_height, std::string last_layer)
      : name_(name),
        type_(type),
        model_desc_path_(model_desc_path),
        model_params_path_(model_params_path),
        input_width_(input_width),
        input_height_(input_height),
        last_layer_(last_layer) {}

  const string& GetName() const { return name_; }
  const ModelType& GetModelType() const { return type_; }
  void SetModelDescPath(const string& file_path) {
    model_desc_path_ = file_path;
  }
  const string& GetModelDescPath() const { return model_desc_path_; }
  const string& GetModelParamsPath() const { return model_params_path_; }
  int GetInputWidth() const { return input_width_; }
  int GetInputHeight() const { return input_height_; }
  const std::string& GetLastLayer() const { return last_layer_; }

  void SetLabelFilePath(const string& file_path) {
    label_file_path_ = file_path;
  }
  const string& GetLabelFilePath() const { return label_file_path_; }

 private:
  string name_;
  ModelType type_;
  string model_desc_path_;
  string model_params_path_;
  int input_width_;
  int input_height_;
  std::string last_layer_;
  // Optional attributes
  string label_file_path_;
};

/**
 * @brief A class representing a DNN model.
 */
class Model {
 public:
  Model(const ModelDesc& model_desc, Shape input_shape);
  ModelDesc GetModelDesc() const;
  virtual void Load() = 0;
  // Convenience function to automatically use the last layer
  std::unordered_map<std::string, cv::Mat> Evaluate(cv::Mat input);
  // Feed the input to the network, run forward, then copy the output from the
  // network
  virtual std::unordered_map<std::string, cv::Mat> Evaluate(
      cv::Mat input, const std::vector<std::string>& output_layer_names) = 0;

 protected:
  ModelDesc model_desc_;
  Shape input_shape_;
};

#endif  // STREAMER_MODEL_MODEL_H_
