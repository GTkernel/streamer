
#include "processor_factory.h"

#include "camera/camera_manager.h"
#include "processor/binary_file_writer.h"
#ifdef USE_CAFFE
#include "processor/caffe_facenet.h"
#endif  // USE_CAFFE
#include "processor/compressor.h"
#include "processor/db_writer.h"
#include "processor/display.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/frame_writer.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/jpeg_writer.h"
#ifdef USE_NCS
#include "processor/detectors/ncs_yolo_detector.h"
#endif  // USE_NCS
#include "processor/detectors/object_detector.h"
#include "processor/detectors/opencv_people_detector.h"
#include "processor/face_tracker.h"
#include "processor/neural_net_evaluator.h"
#include "processor/opencv_motion_detector.h"
#include "processor/pubsub/frame_publisher.h"
#include "processor/pubsub/frame_subscriber.h"
#include "processor/trackers/object_tracker.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#include "processor/strider.h"
#include "processor/throttler.h"
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
    case PROCESSOR_TYPE_DB_WRITER:
      return DbWriter::Create(params);
    case PROCESSOR_TYPE_DISPLAY:
      return Display::Create(params);
    case PROCESSOR_TYPE_ENCODER:
      return GstVideoEncoder::Create(params);
#ifdef USE_CAFFE
    case PROCESSOR_TYPE_FACENET:
      return Facenet::Create(params);
#endif  // USE_CAFFE
    case PROCESSOR_TYPE_FLOW_CONTROL_ENTRANCE:
      return FlowControlEntrance::Create(params);
    case PROCESSOR_TYPE_FLOW_CONTROL_EXIT:
      return FlowControlExit::Create(params);
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
    case PROCESSOR_TYPE_OBJECT_TRACKER:
      return ObjectTracker::Create(params);
    case PROCESSOR_TYPE_OBJECT_DETECTOR:
      return ObjectDetector::Create(params);
    case PROCESSOR_TYPE_FACE_TRACKER:
      return FaceTracker::Create(params);
    case PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR:
      return OpenCVMotionDetector::Create(params);
    case PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR:
      return OpenCVPeopleDetector::Create(params);
    case PROCESSOR_TYPE_STRIDER:
      return Strider::Create(params);
    case PROCESSOR_TYPE_THROTTLER:
      return Throttler::Create(params);
    case PROCESSOR_TYPE_INVALID:
      LOG(FATAL) << "Cannot instantiate a Processor of type: "
                 << GetStringForProcessorType(type);
  }

  LOG(FATAL) << "Unhandled ProcessorType: " << GetStringForProcessorType(type);
}
