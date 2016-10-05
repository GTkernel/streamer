//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model_manager.h"
#include "common/common.h"
#include "utils/utils.h"

#ifdef USE_CAFFE
#ifdef USE_FP16
#include "caffe_fp16_model.h"
#else
#include "caffe_model.h"
#endif
#endif

#ifdef USE_GIE
#include "gie_model.h"
#endif

#ifdef USE_MXNET
#include "mxnet_model.h"
#endif

static const string MODEL_TOML_FILENAME = "models.toml";

ModelManager &ModelManager::GetInstance() {
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

  for (auto model_value : model_values) {
    string name = model_value.get<string>("name");
    string type_string = model_value.get<string>("type");
    ModelType type = MODEL_TYPE_INVALID;
    if (type_string == "caffe") {
      type = MODEL_TYPE_CAFFE;
    } else if (type_string == "mxnet") {
      type = MODEL_TYPE_MXNET;
    } else if (type_string == "gie") {
      type = MODEL_TYPE_GIE;
    } else if (type_string == "tensorflow") {
      type = MODEL_TYPE_TENSORFLOW;
    }
    CHECK(type != MODEL_TYPE_INVALID) << "Type " << type_string
                                      << " is not a valid mode type";
    string desc_path = model_value.get<string>("desc_path");
    string params_path = model_value.get<string>("params_path");
    int input_width = model_value.get<int>("input_width");
    int input_height = model_value.get<int>("input_height");

    ModelDesc model_desc(name, type, desc_path, params_path, input_width,
                         input_height);

    auto label_file_value = model_value.find("label_file");
    if (label_file_value != nullptr) {
      model_desc.SetLabelFilePath(label_file_value->as<string>());
    }

    model_descs_.emplace(name, model_desc);
  }
}

std::vector<int> ModelManager::GetMeanColors() const { return mean_colors_; }
std::unordered_map<string, ModelDesc> ModelManager::GetModelDescs() const {
  return model_descs_;
}
ModelDesc ModelManager::GetModelDesc(const string &name) const {
  auto itr = model_descs_.find(name);
  CHECK(itr != model_descs_.end()) << "Model description with name " << name
                                   << " is not present";
  return itr->second;
}

bool ModelManager::HasModel(const string &name) const {
  return model_descs_.count(name) != 0;
}

std::unique_ptr<Model> ModelManager::CreateModel(const ModelDesc &model_desc,
                                                 Shape input_shape) {
  std::unique_ptr<Model> result;
  if (model_desc.GetModelType() == MODEL_TYPE_CAFFE) {
#ifdef USE_CAFFE
#ifdef USE_FP16
    result.reset(new CaffeFp16Model(model_desc, input_shape));
#else
    result.reset(new CaffeModel<float>(model_desc, input_shape));
#endif
#else
    LOG(FATAL) << "Not build with Caffe, failed to initialize classifier";
#endif
  }

  if (model_desc.GetModelType() == MODEL_TYPE_GIE) {
#ifdef USE_GIE
    result.reset(new GIEModel(model_desc, input_shape));
#else
    LOG(FATAL) << "Not build with GIE, failed to initialize classifier";
#endif
  }

  if (model_desc.GetModelType() == MODEL_TYPE_MXNET) {
#ifdef USE_MXNET
    result.reset(new MXNetModel(model_desc, input_shape));
#else
    LOG(FATAL) << "Not build with MXNet, failed to initialize classifier";
#endif
  }
  return result;
}
