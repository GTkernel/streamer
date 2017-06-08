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
#include "video/gst_video_encoder.h"
#ifdef USE_RPC
#include "rpc/frame_receiver.h"
#include "rpc/frame_sender.h"
#endif
#ifdef USE_ZMQ
#include "stream_publisher.h"
#endif

#include "utils/string_utils.h"

std::shared_ptr<Processor> ProcessorFactory::CreateInstance(
    ProcessorType processor_type, FactoryParamsType params) {
  std::shared_ptr<Processor> processor;

  switch (processor_type) {
    case PROCESSOR_TYPE_CAMERA:
      processor = std::dynamic_pointer_cast<Processor>(CreateCamera(params));
      break;
    case PROCESSOR_TYPE_CUSTOM:
      processor.reset(CreateCustomProcessor(params));
      break;
    case PROCESSOR_TYPE_DUMMY_NN:
      processor.reset(CreateDummyNNProcessor(params));
      break;
    case PROCESSOR_TYPE_ENCODER:
      processor.reset(CreateEncoder(params));
      break;
    case PROCESSOR_TYPE_FILE_WRITER:
      processor.reset(CreateFileWriter(params));
      break;
#ifdef USE_RPC
    case PROCESSOR_TYPE_FRAME_RECEIVER:
      processor.reset(CreateFrameReceiver(params));
      break;
    case PROCESSOR_TYPE_FRAME_SENDER:
      processor.reset(CreateFrameSender(params));
      break;
#endif
    case PROCESSOR_TYPE_IMAGE_CLASSIFIER:
      processor.reset(CreateImageClassifier(params));
      break;
    case PROCESSOR_TYPE_IMAGE_SEGMENTER:
      processor.reset(CreateImageSegmenter(params));
      break;
    case PROCESSOR_TYPE_IMAGE_TRANSFORMER:
      processor.reset(CreateImageTransformer(params));
      break;
    case PROCESSOR_TYPE_NEURAL_NET_EVALUATOR:
      processor.reset(CreateNeuralNetEvaluator(params));
      break;
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      processor.reset(CreateOpenCVFaceDetector(params));
      break;
#ifdef USE_ZMQ
    case PROCESSOR_TYPE_STREAM_PUBLISHER:
      processor.reset(CreateStreamPublisher(params));
      break;
#endif
    default:
      LOG(FATAL) << "Unknown processor type";
  }

  return processor;
}

std::shared_ptr<Camera> ProcessorFactory::CreateCamera(
    const FactoryParamsType& params) {
  string camera_name = params.at("camera_name");
  auto& camera_manager = CameraManager::GetInstance();
  if (camera_manager.HasCamera(camera_name)) {
    return camera_manager.GetCamera(camera_name);
  } else {
    return nullptr;
  }
}

Processor* ProcessorFactory::CreateCustomProcessor(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

Processor* ProcessorFactory::CreateDummyNNProcessor(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

Processor* ProcessorFactory::CreateEncoder(const FactoryParamsType& params) {
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

  GstVideoEncoder* encoder;
  if (port > 0) {
    encoder = new GstVideoEncoder(width, height, port);
  } else {
    encoder = new GstVideoEncoder(width, height, filename);
  }

  return encoder;
}

Processor* ProcessorFactory::CreateFileWriter(const FactoryParamsType& params) {
  return new FileWriter(params.at("filename"));
}

#ifdef USE_RPC
Processor* ProcessorFactory::CreateFrameReceiver(
    const FactoryParamsType& params) {
  return new FrameReceiver(params.at("list_url"));
};

Processor* ProcessorFactory::CreateFrameSender(
    const FactoryParamsType& params) {
  return new FrameSender(params.at("server_url"));
};
#endif

Processor* ProcessorFactory::CreateImageClassifier(
    const FactoryParamsType& params) {
  ModelManager& model_manager = ModelManager::GetInstance();
  std::string model_name = params.at("model");
  CHECK(model_manager.HasModel(model_name));
  ModelDesc model_desc = model_manager.GetModelDesc(model_name);
  size_t num_labels = StringToSizet(params.at("num_labels"));

  auto num_channels_pair = params.find("num_channels");
  if (num_channels_pair == params.end()) {
    return new ImageClassifier(model_desc, num_labels);
  } else {
    // If num_channels is specified, then we need to use the constructor that
    // creates a hidden NeuralNetEvaluator.
    size_t num_channels = StringToSizet(num_channels_pair->second);
    Shape input_shape = Shape(num_channels, model_desc.GetInputWidth(),
                              model_desc.GetInputHeight());
    return new ImageClassifier(model_desc, input_shape, num_labels);
  }
}

Processor* ProcessorFactory::CreateImageSegmenter(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

Processor* ProcessorFactory::CreateImageTransformer(
    const FactoryParamsType& params) {
  int width = atoi(params.at("width").c_str());
  int height = atoi(params.at("height").c_str());
  // Default channel = 3
  int channel = 3;
  if (params.count("channel") != 0)
    channel = atoi(params.at("channel").c_str());

  return new ImageTransformer(Shape(channel, width, height), true);
}

Processor* ProcessorFactory::CreateNeuralNetEvaluator(
    const FactoryParamsType& params) {
  ModelManager& model_manager = ModelManager::GetInstance();
  std::string model_name = params.at("model");
  CHECK(model_manager.HasModel(model_name));
  ModelDesc model_desc = model_manager.GetModelDesc(model_name);

  size_t num_channels = StringToSizet(params.at("num_channels"));
  Shape input_shape = Shape(num_channels, model_desc.GetInputWidth(),
                            model_desc.GetInputHeight());

  std::vector<std::string> output_layer_names = {
      params.at("output_layer_name")};
  return new NeuralNetEvaluator(model_desc, input_shape, output_layer_names);
}

Processor* ProcessorFactory::CreateOpenCVFaceDetector(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

#ifdef USE_ZMQ
Processor* ProcessorFactory::CreateStreamPublisher(
    const FactoryParamsType& params) {
  if (params.count("listen_url") != 0) {
    auto url = params.at("listen_url");
    return new StreamPublisher(params.at("name"), url);
  }
  return new StreamPublisher(params.at("name"));
}
#endif
