//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model_manager.h"
#include "common/common.h"
#include "common/context.h"
#include "utils/utils.h"

#ifdef USE_CAFFE
#include "caffe_model.h"
#endif  // USE_CAFFE

static const string MODEL_TOML_FILENAME = "models.toml";

ModelManager& ModelManager::GetInstance() {
  static ModelManager manager;
  return manager;
}

ModelManager::ModelManager() {
  // FIXME: Use a safer way to construct path.
  string model_toml_path =
      Context::GetContext().GetConfigFile(MODEL_TOML_FILENAME);
  auto root_value = ParseTomlFromFile(model_toml_path);
  // Get mean colors
  auto mean_image_value = root_value.find("mean_image");
  CHECK(mean_image_value != nullptr) << "[mean_image] is not found";
  int mean_blue = mean_image_value->get<int>("BLUE");
  int mean_green = mean_image_value->get<int>("GREEN");
  int mean_red = mean_image_value->get<int>("RED");
  mean_colors_ = {mean_blue, mean_green, mean_red};

  // Get model descriptions
  auto model_values = root_value.find("model")->as<toml::Array>();

  for (const auto& model_value : model_values) {
    string name = model_value.get<string>("name");
    string type_string = model_value.get<string>("type");
    ModelType type = MODEL_TYPE_INVALID;
    if (type_string == "caffe") {
      type = MODEL_TYPE_CAFFE;
    } else if (type_string == "tensorflow") {
      type = MODEL_TYPE_TENSORFLOW;
    } else if (type_string == "ncs") {
      type = MODEL_TYPE_NCS;
    }
    CHECK(type != MODEL_TYPE_INVALID)
        << "Type " << type_string << " is not a valid mode type";
    std::vector<string> desc_paths;
    std::vector<string> params_paths;
    auto desc_path = model_value.find("desc_path");
    if (desc_path->is<toml::Array>()) {
      auto desc_path_array = desc_path->as<toml::Array>();
      for (const auto& m : desc_path_array) {
        desc_paths.push_back(m.as<std::string>());
      }
    } else {
      string desc_path_str = desc_path->as<string>();
      desc_paths.push_back(desc_path_str);
    }
    auto params_path = model_value.find("params_path");
    if (params_path->is<toml::Array>()) {
      auto params_path_array = params_path->as<toml::Array>();
      for (const auto& m : params_path_array) {
        params_paths.push_back(m.as<std::string>());
      }
    } else {
      string params_path_str = params_path->as<string>();
      params_paths.push_back(params_path_str);
    }
    CHECK(desc_paths.size() == params_paths.size());
    // string desc_path = model_value.get<string>("desc_path");
    // string params_path = model_value.get<string>("params_path");
    int input_width = model_value.get<int>("input_width");
    int input_height = model_value.get<int>("input_height");

    //    CHECK(model_value.has("last_layer"))
    //        << "Model \"" << name << "\" is missing the \"last_layer\"
    //        parameter!";
    //    std::string last_layer = model_value.get<std::string>("last_layer");

    std::string last_layer = "prob";
    if (model_value.has("last_layer")) {
      last_layer = model_value.get<std::string>("last_layer");
    }

    std::vector<ModelDesc> model_descs;
    for (size_t i = 0; i < desc_paths.size(); ++i) {
      ModelDesc model_desc(name, type, desc_paths.at(i), params_paths.at(i),
                           input_width, input_height, last_layer);

      auto label_file_value = model_value.find("label_file");
      if (label_file_value != nullptr) {
        model_desc.SetLabelFilePath(label_file_value->as<string>());
      }
      auto voc_config_value = model_value.find("voc_config");
      if (voc_config_value != nullptr) {
        model_desc.SetVocConfigPath(voc_config_value->as<string>());
      }

      model_descs.push_back(model_desc);
    }
    model_descs_.emplace(name, model_descs);
  }
}

std::vector<int> ModelManager::GetMeanColors() const { return mean_colors_; }
// std::unordered_map<string, ModelDesc> ModelManager::GetModelDescs() const {
//  return model_descs_;
//}
ModelDesc ModelManager::GetModelDesc(const string& name) const {
  auto itr = model_descs_.find(name);
  CHECK(itr != model_descs_.end())
      << "Model description with name " << name << " is not present";
  return itr->second.at(0);
}

std::vector<ModelDesc> ModelManager::GetModelDescs(const string& name) const {
  auto itr = model_descs_.find(name);
  CHECK(itr != model_descs_.end())
      << "Model description with name " << name << " is not present";
  return itr->second;
}

bool ModelManager::HasModel(const string& name) const {
  return model_descs_.count(name) != 0;
}

std::unique_ptr<Model> ModelManager::CreateModel(const ModelDesc& model_desc,
                                                 Shape input_shape,
                                                 size_t batch_size) {
  // Silence compiler warnings when none of the #ifdefs below are active
  (void)input_shape;
  (void)batch_size;

  std::unique_ptr<Model> result;
  if (model_desc.GetModelType() == MODEL_TYPE_CAFFE) {
#ifdef USE_CAFFE
    result.reset(new CaffeModel<float>(model_desc, input_shape));
#else
    LOG(FATAL) << "Not built with Caffe, failed to initialize classifier";
#endif  // USE_CAFFE
  }
  return result;
}
