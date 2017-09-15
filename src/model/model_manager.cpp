//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model/model_manager.h"

#include <stdexcept>

#include "common/common.h"
#include "common/context.h"
#ifdef USE_CAFFE
#include "model/caffe_model.h"
#endif  // USE_CAFFE
#ifdef USE_TENSORFLOW
#include "model/tf_model.h"
#endif  // USE_TENSORFLOW
#include "utils/utils.h"

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
  double mean_blue = mean_image_value->get<double>("BLUE");
  double mean_green = mean_image_value->get<double>("GREEN");
  double mean_red = mean_image_value->get<double>("RED");
  mean_colors_ = cv::Scalar(mean_blue, mean_green, mean_red);

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
    } else if (type_string == "opencv") {
      type = MODEL_TYPE_OPENCV;
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

    if (model_value.has("params_path")) {
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
    } else {
      params_paths.push_back("");
    }
    CHECK(desc_paths.size() == params_paths.size());
    int input_width = model_value.get<int>("input_width");
    int input_height = model_value.get<int>("input_height");

    std::string default_output_layer = "";
    if (model_value.has("default_output_layer")) {
      default_output_layer =
          model_value.get<std::string>("default_output_layer");
    }
    std::string default_input_layer;
    if (type_string == "tensorflow") {
      CHECK(model_value.has("default_input_layer"))
          << "Model \"" << name
          << "\" is missing the \"default_input_layer\" parameter!";
      default_input_layer = model_value.get<std::string>("default_input_layer");
    } else if (type_string == "caffe" &&
               model_value.has("default_input_layer")) {
      LOG(WARNING) << "Caffe does not support specifying an input layer. "
                   << "Ignoring \"default_input_layer\" param.";
      default_input_layer = "";
    }

    auto input_scale_value = model_value.find("input_scale");
    if (input_scale_value != nullptr) {
      if (type_string != "caffe") {
        LOG(WARNING)
            << "Only Caffe models support specifying an input scale factor. "
            << "Ignoring \"input_scale\" param.";
      }
    }

    std::vector<ModelDesc> model_descs;
    for (size_t i = 0; i < desc_paths.size(); ++i) {
      ModelDesc model_desc(name, type, desc_paths.at(i), params_paths.at(i),
                           input_width, input_height, default_input_layer,
                           default_output_layer);

      auto label_file_value = model_value.find("label_file");
      if (label_file_value != nullptr) {
        model_desc.SetLabelFilePath(label_file_value->as<string>());
      }
      auto voc_config_value = model_value.find("voc_config");
      if (voc_config_value != nullptr) {
        model_desc.SetVocConfigPath(voc_config_value->as<string>());
      }
      if ((input_scale_value != nullptr) && (type_string == "caffe")) {
        model_desc.SetInputScale(input_scale_value->as<double>());
      }

      model_descs.push_back(model_desc);
    }
    model_descs_.emplace(name, model_descs);
  }
}

cv::Scalar ModelManager::GetMeanColors() const { return mean_colors_; }

void ModelManager::SetMeanColors(cv::Scalar mean_colors) {
  mean_colors_ = mean_colors;
}

std::unordered_map<std::string, std::vector<ModelDesc>>
ModelManager::GetAllModelDescs() const {
  return model_descs_;
}

ModelDesc ModelManager::GetModelDesc(const string& name) const {
  return GetModelDescs(name).at(0);
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

  ModelType model_type = model_desc.GetModelType();
  switch (model_type) {
    case MODEL_TYPE_INVALID:
      throw std::logic_error("Cannot create a model for MODEL_TYPE_INVALID.");
    case MODEL_TYPE_CAFFE:
#ifdef USE_CAFFE
      return std::make_unique<CaffeModel<float>>(model_desc, input_shape, batch_size);
#else
      throw std::logic_error(
          "Not built with Caffe. Failed to initialize model!");
#endif  // USE_CAFFE
    case MODEL_TYPE_OPENCV:
      STREAMER_NOT_IMPLEMENTED;
      return nullptr;
    case MODEL_TYPE_NCS:
#ifdef USE_NCS
      STREAMER_NOT_IMPLEMENTED;
      return nullptr;
#else
      throw std::logic_error("Not built with NCS. Failed to initialize model!");
#endif  // USE_NCS
    case MODEL_TYPE_TENSORFLOW:
#ifdef USE_TENSORFLOW
      return std::make_unique<TFModel>(model_desc, input_shape);
#else
      throw std::logic_error(
          "Not built with TensorFlow. Failed to initialize model!");
#endif  // USE_TENSORFLOW
  }

  throw std::runtime_error("Unhandled ModelType: " +
                           std::to_string(model_type));
}
