//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#ifndef STREAMER_MODEL_H
#define STREAMER_MODEL_H

#include <common/data_buffer.h>
#include "common/common.h"
#include "model.h"

/**
 * @brief A decription of a DNN model, created from models.toml file. A
 * ModelDesc can be used to initialize a model.
 */
class ModelDesc {
 public:
  ModelDesc() {}
  ModelDesc(const string &name, const ModelType &type,
            const string &model_desc_path, const string &model_params_path,
            int input_width, int input_height)
      : name_(name),
        type_(type),
        model_desc_path_(model_desc_path),
        model_params_path_(model_params_path),
        input_width_(input_width),
        input_height_(input_height) {}

  const string &GetName() const { return name_; }
  const ModelType &GetModelType() const { return type_; }
  const string &GetModelDescPath() const { return model_desc_path_; }
  const string &GetModelParamsPath() const { return model_params_path_; }
  int GetInputWidth() const { return input_width_; }
  int GetInputHeight() const { return input_height_; }

  void SetLabelFilePath(const string &file_path) {
    label_file_path_ = file_path;
  }
  const string &GetLabelFilePath() const { return label_file_path_; }
  void SetVocConfigPath(const string &file_path) {
    voc_config_path_ = file_path;
  }
  const string& GetVocConfigPath() const { return voc_config_path_; }

 private:
  string name_;
  ModelType type_;
  string model_desc_path_;
  string model_params_path_;
  int input_width_;
  int input_height_;
  // Optional attributes
  string label_file_path_;
  string voc_config_path_;
};

class ModelDescription {
 public:
  ModelDescription() {}
  ModelDescription(const string &name, const ModelType &type, const toml::Value& value)
      : name_(name),
        type_(type),
        value_(value) {}

  const string &GetName() const { return name_; }
  const ModelType &GetModelType() const { return type_; }
  toml::Value& operator[](const std::string& key) { return value_[key]; }

 private:
  string name_;
  ModelType type_;
  toml::Value value_;
};

/**
 * @brief A class representing a DNN model.
 */
class Model {
 public:
  Model(const ModelDesc &model_desc, Shape input_shape, size_t batch_size = 1);
  ModelDesc GetModelDesc() const;
  virtual void Load() = 0;
  // Feed the input to the network, run forward, then copy the output from the
  // network
  virtual void Evaluate() = 0;
  // Run pure forward pass, copy no input or ouput, this is only supposed to be
  // used by experiment.
  virtual void Forward() = 0;
  DataBuffer GetInputBuffer();
  std::vector<DataBuffer> GetOutputBuffers();
  std::vector<Shape> GetOutputShapes();

 protected:
  ModelDesc model_desc_;
  Shape input_shape_;
  DataBuffer input_buffer_;
  std::vector<DataBuffer> output_buffers_;
  std::vector<Shape> output_shapes_;
  size_t batch_size_;
};

#endif  // STREAMER_MODEL_H
