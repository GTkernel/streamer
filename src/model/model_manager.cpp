//
// Created by xianran on 9/24/16.
//

#include "model_manager.h"

static const string MODEL_TOML_PATH = "config/models.toml";

ModelManager &ModelManager::GetInstance() {
  static ModelManager manager;
  return manager;
}

ModelManager::ModelManager() {
  auto root_value = ParseTomlFromFile(MODEL_TOML_PATH);
  // Get mean colors
  auto mean_image_value = root_value.find("mean_image");
  CHECK(mean_image_value != nullptr) << "[mean_image] is not found";
  int mean_blue = mean_image_value->get<int>("BLUE");
  int mean_green = mean_image_value->get<int>("GREEN");
  int mean_red = mean_image_value->get<int>("RED");
  mean_colors_ = {mean_blue, mean_green, mean_red};

  // Get model descriptions
  auto model_values = root_value.find("model")->as<toml::Array>();
  
  for (auto model_value : model_values) {
    string name = model_value.get<string>("name");
    string type_string = model_value.get<string>("type");
    ModelType type = MODEL_TYPE_INVALID;
    if (type_string == "caffe") {
      type = MODEL_TYPE_CAFFE;
    }
    CHECK(type != MODEL_TYPE_INVALID)
    << "Type " << type_string << " is not a valid mode type";
    string desc_path = model_value.get<string>("desc_path");
    string params_path = model_value.get<string>("params_path");
    int input_width = model_value.get<int>("input_width");
    int input_height = model_value.get<int>("input_height");

    model_descs_.emplace(name,
                         ModelDesc(name,
                                   type,
                                   desc_path,
                                   params_path,
                                   input_width,
                                   input_height));
  }
}

std::vector<int> ModelManager::GetMeanColors() const {
  return mean_colors_;
}
std::unordered_map<string, ModelDesc> ModelManager::GetModelDescs() const {
  return model_descs_;
}
ModelDesc ModelManager::GetModelDesc(const string &name) const {
  auto itr = model_descs_.find(name);
  CHECK(itr != model_descs_.end())
  << "Model description with name " << name << " is not present";
  return itr->second;
}
