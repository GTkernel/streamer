//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "processor_factory.h"

#include "camera/camera_manager.h"
#include "processor/dummy_nn_processor.h"
#include "processor/file_writer.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/neural_net_evaluator.h"
#include "processor/opencv_face_detector.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#ifdef USE_ZMQ
#include "processor/stream_publisher.h"
#endif  // USE_ZMQ
#include "video/gst_video_encoder.h"

std::shared_ptr<Processor> ProcessorFactory::Create(
    ProcessorType processor_type, FactoryParamsType params) {
  switch (processor_type) {
    case PROCESSOR_TYPE_CAMERA:
      return CameraManager::GetInstance().GetCamera(params.at("camera_name"));
    case PROCESSOR_TYPE_CUSTOM:
      STREAMER_NOT_IMPLEMENTED;
      return nullptr;
    case PROCESSOR_TYPE_DUMMY_NN:
      return DummyNNProcessor::Create(params);
    case PROCESSOR_TYPE_ENCODER:
      return GstVideoEncoder::Create(params);
    case PROCESSOR_TYPE_FILE_WRITER:
      return FileWriter::Create(params);
#ifdef USE_RPC
    case PROCESSOR_TYPE_FRAME_RECEIVER:
      return FrameReceiver::Create(params);
    case PROCESSOR_TYPE_FRAME_SENDER:
      return FrameSender::Create(params);
#endif  // USE_RPC
    case PROCESSOR_TYPE_IMAGE_CLASSIFIER:
      return ImageClassifier::Create(params);
    case PROCESSOR_TYPE_IMAGE_SEGMENTER:
      return ImageSegmenter::Create(params);
    case PROCESSOR_TYPE_IMAGE_TRANSFORMER:
      return ImageTransformer::Create(params);
    case PROCESSOR_TYPE_NEURAL_NET_EVALUATOR:
      return NeuralNetEvaluator::Create(params);
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      return OpenCVFaceDetector::Create(params);
#ifdef USE_ZMQ
    case PROCESSOR_TYPE_STREAM_PUBLISHER:
      return StreamPublisher::Create(params);
#endif  // USE_ZMQ
    default:
      LOG(FATAL) << "Unknown processor type.";
  }
}
