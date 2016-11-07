//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "processor_factory.h"
#include "camera/camera_manager.h"
#include "model/model_manager.h"

std::shared_ptr<Processor> ProcessorFactory::CreateInstance(
    ProcessorType processor_type, FactoryParamsType params) {
  std::shared_ptr<Processor> processor;
  switch (processor_type) {
    case PROCESSOR_TYPE_CUSTOM:
      processor.reset(CreateCustomProcessor(params));
      break;
    case PROCESSOR_TYPE_CAMERA:
      processor = std::dynamic_pointer_cast<Processor>(CreateCamera(params));
      break;
    case PROCESSOR_TYPE_DUMMY_NN:
      processor.reset(CreateDummyNNProcessor(params));
      break;
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      processor.reset(CreateOpenCVFaceDetector(params));
      break;
    case PROCESSOR_TYPE_IMAGE_TRANSFORMER:
      processor.reset(CreateImageTransformer(params));
      break;
    case PROCESSOR_TYPE_IMAGE_CLASSIFIER:
      processor.reset(CreateImageClassifier(params));
      break;
    case PROCESSOR_TYPE_IMAGE_SEGMENTER:
      processor.reset(CreateImageSegmenter(params));
      break;
    case PROCESSOR_TYPE_ENCODER:
      processor.reset(CreateEncoder(params));
      break;
    default:
      LOG(FATAL) << "Unknown processor type";
  }

  return processor;
}

Processor *ProcessorFactory::CreateCustomProcessor(
    const FactoryParamsType &params) {
  return nullptr;
}
std::shared_ptr<Camera> ProcessorFactory::CreateCamera(
    const FactoryParamsType &params) {
  string camera_name = params.at("camera_name");
  auto &camera_manager = CameraManager::GetInstance();
  if (camera_manager.HasCamera(camera_name)) {
    return camera_manager.GetCamera(camera_name);
  } else {
    return nullptr;
  }
}
Processor *ProcessorFactory::CreateEncoder(const FactoryParamsType &params) {
  return nullptr;
}
Processor *ProcessorFactory::CreateImageClassifier(
    const FactoryParamsType &params) {
  auto &model_manager = ModelManager::GetInstance();
  if (model_manager.HasModel(params.at("model"))) {
    auto model_desc = model_manager.GetModelDesc(params.at("model"));
    return new ImageClassifier(
        model_desc,
        Shape(model_desc.GetInputWidth(), model_desc.GetInputHeight()), 1);
  } else {
    return nullptr;
  }
}
Processor *ProcessorFactory::CreateImageSegmenter(
    const FactoryParamsType &params) {
  return nullptr;
}
Processor *ProcessorFactory::CreateImageTransformer(
    const FactoryParamsType &params) {
  int width = atoi(params.at("width").c_str());
  int height = atoi(params.at("height").c_str());
  // Default channel = 3
  int channel = 3;
  if (params.count("channel") != 0)
    channel = atoi(params.at("channel").c_str());

  return new ImageTransformer(Shape(channel, width, height), CROP_TYPE_CENTER,
                              true);
}
Processor *ProcessorFactory::CreateOpenCVFaceDetector(
    const FactoryParamsType &params) {
  return nullptr;
}
Processor *ProcessorFactory::CreateDummyNNProcessor(
    const FactoryParamsType &params) {
  return nullptr;
}
