//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "processor_factory.h"
#include "camera/camera_manager.h"
#include "dummy_nn_processor.h"
#include "file_writer.h"
#include "image_classifier.h"
#include "image_segmenter.h"
#include "image_transformer.h"
#include "model/model_manager.h"
#include "opencv_face_detector.h"
#include "stream_publisher.h"
#include "video/gst_video_encoder.h"

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
    case PROCESSOR_TYPE_STREAM_PUBLISHER:
      processor.reset(CreateStreamPublisher(params));
      break;
    case PROCESSOR_TYPE_FILE_WRITER:
      processor.reset(CreateFileWriter(params));
      break;
    default:
      LOG(FATAL) << "Unknown processor type";
  }

  return processor;
}

Processor *ProcessorFactory::CreateCustomProcessor(
    const FactoryParamsType &) {
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
  int port = -1;
  string filename;

  if (params.count("port") != 0) {
    port = atoi(params.at("port").c_str());
  } else if (params.count("filename") != 0) {
    filename = params.at("filename");
  } else {
    LOG(FATAL) << "At least port or filename is needed for encoder";
  }

  int width = atoi(params.at("width").c_str());
  int height = atoi(params.at("height").c_str());

  GstVideoEncoder *encoder;
  if (port > 0) {
    encoder = new GstVideoEncoder(width, height, port);
  } else {
    encoder = new GstVideoEncoder(width, height, filename);
  }

  return encoder;
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
    const FactoryParamsType &) {
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

  return new ImageTransformer(Shape(channel, width, height), true);
}
Processor *ProcessorFactory::CreateOpenCVFaceDetector(
    const FactoryParamsType &) {
  return nullptr;
}
Processor *ProcessorFactory::CreateDummyNNProcessor(
    const FactoryParamsType &) {
  return nullptr;
}

Processor *ProcessorFactory::CreateStreamPublisher(
    const FactoryParamsType &params) {
  return new StreamPublisher(params.at("name"));
}

Processor *ProcessorFactory::CreateFileWriter(const FactoryParamsType &params) {
  return new FileWriter(params.at("filename"));
}
