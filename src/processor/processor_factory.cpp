
#include "processor_factory.h"

#include "camera/camera_manager.h"
#include "processor/binary_file_writer.h"
#include "processor/caffe_mtcnn.h"
#include "processor/compressor.h"
#include "processor/db_writer.h"
#include "processor/facenet.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/frame_writer.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/jpeg_writer.h"
#ifdef USE_NCS
#include "processor/ncs_yolo_detector.h"
#endif  // USE_NCS
#include "processor/neural_net_evaluator.h"
#include "processor/obj_tracker.h"
#ifdef USE_FRCNN
#include "processor/object_detector.h"
#endif  // USE_FRCNN
#include "processor/object_tracker.h"
#include "processor/opencv_face_detector.h"
#include "processor/opencv_motion_detector.h"
#include "processor/opencv_people_detector.h"
#include "processor/pubsub/frame_publisher.h"
#include "processor/pubsub/frame_subscriber.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#ifdef USE_SSD
#include "processor/ssd_detector.h"
#endif  // USE_SSD
#include "processor/throttler.h"
#include "processor/yolo_detector.h"
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
    case PROCESSOR_TYPE_ENCODER:
      return GstVideoEncoder::Create(params);
    case PROCESSOR_TYPE_FACENET:
      return Facenet::Create(params);
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
    case PROCESSOR_TYPE_MTCNN_FACE_DETECTOR:
      return MtcnnFaceDetector::Create(params);
#ifdef USE_NCS
    case PROCESSOR_TYPE_NCS_YOLO_DETECTOR:
      return NcsYoloDetector::Create(params);
#endif  // USE_NCS
    case PROCESSOR_TYPE_NEURAL_NET_EVALUATOR:
      return NeuralNetEvaluator::Create(params);
    case PROCESSOR_TYPE_OBJ_TRACKER:
      return ObjTracker::Create(params);
#ifdef USE_FRCNN
    case PROCESSOR_TYPE_OBJECT_DETECTOR:
      return ObjectDetector::Create(params);
#endif  // USE_FRCNN
    case PROCESSOR_TYPE_OBJECT_TRACKER:
      return ObjectTracker::Create(params);
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      return OpenCVFaceDetector::Create(params);
    case PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR:
      return OpenCVMotionDetector::Create(params);
    case PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR:
      return OpenCVPeopleDetector::Create(params);
#ifdef USE_SSD
    case PROCESSOR_TYPE_SSD_DETECTOR:
      return SsdDetector::Create(params);
#endif  // USE_SSD
    case PROCESSOR_TYPE_THROTTLER:
      return Throttler::Create(params);
    case PROCESSOR_TYPE_YOLO_DETECTOR:
      return YoloDetector::Create(params);
    case PROCESSOR_TYPE_INVALID:
      LOG(FATAL) << "Cannot instantiate a Processor of type: "
                 << GetStringForProcessorType(type);
  }

  LOG(FATAL) << "Unhandled ProcessorType: " << GetStringForProcessorType(type);
}
