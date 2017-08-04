
#include "processor_factory.h"

#include "camera/camera_manager.h"
#include "processor/binary_file_writer.h"
#include "processor/compressor.h"
#include "processor/frame_writer.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/jpeg_writer.h"
#include "processor/neural_net_evaluator.h"
#include "processor/opencv_face_detector.h"
#include "processor/throttler.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#include "processor/pubsub/frame_publisher.h"
#include "processor/pubsub/frame_subscriber.h"
#include "video/gst_video_encoder.h"

std::shared_ptr<Processor> ProcessorFactory::Create(ProcessorType type,
                                                    FactoryParamsType params) {
  switch (type) {
    case PROCESSOR_TYPE_BINARY_FILE_WRITER:
      return BinaryFileWriter::Create(params);
    case PROCESSOR_TYPE_CAMERA:
      return CameraManager::GetInstance().GetCamera(params.at("camera_name"));
    case PROCESSOR_TYPE_COMPRESSOR:
      return Compressor::Create(params);
    case PROCESSOR_TYPE_CUSTOM:
      STREAMER_NOT_IMPLEMENTED;
      return nullptr;
    case PROCESSOR_TYPE_ENCODER:
      return GstVideoEncoder::Create(params);
#ifdef USE_RPC
    case PROCESSOR_TYPE_FRAME_RECEIVER:
      return FrameReceiver::Create(params);
    case PROCESSOR_TYPE_FRAME_SENDER:
      return FrameSender::Create(params);
#endif  // USE_RPC
    case PROCESSOR_TYPE_FRAME_PUBLISHER:
      return FramePublisher::Create(params);
    case PROCESSOR_TYPE_FRAME_SUBSCRIBER:
      return FrameSubscriber::Create(params);
    case PROCESSOR_TYPE_FRAME_WRITER:
      return FrameWriter::Create(params);
    case PROCESSOR_TYPE_IMAGE_CLASSIFIER:
      return ImageClassifier::Create(params);
    case PROCESSOR_TYPE_IMAGE_SEGMENTER:
      return ImageSegmenter::Create(params);
    case PROCESSOR_TYPE_IMAGE_TRANSFORMER:
      return ImageTransformer::Create(params);
    case PROCESSOR_TYPE_JPEG_WRITER:
      return JpegWriter::Create(params);
    case PROCESSOR_TYPE_NEURAL_NET_EVALUATOR:
      return NeuralNetEvaluator::Create(params);
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      return OpenCVFaceDetector::Create(params);
    case PROCESSOR_TYPE_THROTTLER:
      return Throttler::Create(params);
    case PROCESSOR_TYPE_INVALID:
      LOG(FATAL) << "Cannot instantiate a Processor of type: "
                 << GetStringForProcessorType(type);
  }

  LOG(FATAL) << "Unhandled ProcessorType: " << GetStringForProcessorType(type);
}
