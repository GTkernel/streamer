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
    }
    CHECK(type != MODEL_TYPE_INVALID)
        << "Type " << type_string << " is not a valid mode type";
    string desc_path = model_value.get<string>("desc_path");
    string params_path;
    if (model_value.has("params_path")) {
      params_path = model_value.get<string>("params_path");
    } else {
      params_path = "";
    }
    int input_width = model_value.get<int>("input_width");
    int input_height = model_value.get<int>("input_height");

    CHECK(model_value.has("default_output_layer"))
        << "Model \"" << name
        << "\" is missing the \"default_output_layer\" parameter!";
    std::string default_output_layer =
        model_value.get<std::string>("default_output_layer");
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

    ModelDesc model_desc(name, type, desc_path, params_path, input_width,
                         input_height, default_input_layer,
                         default_output_layer);

    auto label_file_value = model_value.find("label_file");
    if (label_file_value != nullptr) {
      model_desc.SetLabelFilePath(label_file_value->as<string>());
    }

    model_descs_.emplace(name, model_desc);
  }
}

cv::Scalar ModelManager::GetMeanColors() const { return mean_colors_; }

void ModelManager::SetMeanColors(cv::Scalar mean_colors) {
  mean_colors_ = mean_colors;
}

std::unordered_map<string, ModelDesc> ModelManager::GetModelDescs() const {
  return model_descs_;
}
ModelDesc ModelManager::GetModelDesc(const string& name) const {
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
      return std::make_unique<CaffeModel<float>>(model_desc, input_shape);
#else
      throw std::logic_error(
          "Not built with Caffe. Failed to initialize model!");
#endif  // USE_CAFFE
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
