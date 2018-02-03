//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_MODEL_H_
#define STREAMER_MODEL_MODEL_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "common/types.h"

#include "model.h"

/**
 * @brief A decription of a DNN model, created from models.toml file. A
 * ModelDesc can be used to initialize a model.
 */
class ModelDesc {
 public:
  ModelDesc() {}
  ModelDesc(const std::string& name, const ModelType& type,
            const std::string& model_desc_path,
            const std::string& model_params_path, int input_width,
            int input_height, std::string default_input_layer,
            std::string default_output_layer)
      : name_(name),
        type_(type),
        model_desc_path_(model_desc_path),
        model_params_path_(model_params_path),
        input_width_(input_width),
        input_height_(input_height),
        default_input_layer_(default_input_layer),
        default_output_layer_(default_output_layer),
        input_scale_(1.0) {}

  const std::string& GetName() const { return name_; }
  const ModelType& GetModelType() const { return type_; }
  const std::string& GetModelDescPath() const { return model_desc_path_; }
  const std::string& GetModelParamsPath() const { return model_params_path_; }
  int GetInputWidth() const { return input_width_; }
  int GetInputHeight() const { return input_height_; }
  const std::string& GetDefaultInputLayer() const {
    return default_input_layer_;
  }
  const std::string& GetDefaultOutputLayer() const {
    return default_output_layer_;
  }
  void SetLabelFilePath(const std::string& file_path) {
    label_file_path_ = file_path;
  }
  const std::string& GetLabelFilePath() const { return label_file_path_; }
  void SetVocConfigPath(const std::string& file_path) {
    voc_config_path_ = file_path;
  }
  const std::string& GetVocConfigPath() const { return voc_config_path_; }
  void SetInputScale(const double& input_scale) { input_scale_ = input_scale; }
  const double& GetInputScale() const { return input_scale_; }

 private:
  std::string name_;
  ModelType type_;
  std::string model_desc_path_;
  std::string model_params_path_;
  int input_width_;
  int input_height_;
  std::string default_input_layer_;
  std::string default_output_layer_;
  // Optional attributes
  std::string label_file_path_;
  std::string voc_config_path_;
  double input_scale_;
};

/**
 * @brief A class representing a DNN model.
 */
class Model {
 public:
  Model(const ModelDesc& model_desc, Shape input_shape, size_t batch_size = 1);
  virtual ~Model();
  ModelDesc GetModelDesc() const;
  virtual void Load() = 0;
  // Convenience function to automatically use the default input and output
  // layers.
  std::unordered_map<std::string, std::vector<cv::Mat>> Evaluate(cv::Mat input);
  // Feed the input to the network, run forward, then copy the output from the
  // network
  virtual std::unordered_map<std::string, std::vector<cv::Mat>> Evaluate(
      const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map,
      const std::vector<std::string>& output_layer_names,
      std::vector<long>* timing_data) = 0;

 protected:
  ModelDesc model_desc_;
  Shape input_shape_;
  size_t batch_size_;
};

#endif  // STREAMER_MODEL_MODEL_H_
