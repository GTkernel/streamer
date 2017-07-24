//
// Created by Ran Xian on 7/26/16.
//

#ifndef STREAMER_COMMON_TYPES_H_
#define STREAMER_COMMON_TYPES_H_

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

#include <boost/serialization/access.hpp>

#include "common/common.h"
#include "common/serialization.h"

/**
 * @brief 3-D shape structure
 */
struct Shape {
  Shape() : channel(0), width(0), height(0){};
  Shape(int c, int w, int h) : channel(c), width(w), height(h){};
  Shape(int w, int h) : channel(1), width(w), height(h){};
  /**
   * @brief Return volumn (size) of the shape object
   */
  size_t GetSize() const { return (size_t)channel * width * height; }
  // Number of channels
  int channel;
  // Width
  int width;
  // Height
  int height;
};

/**
 * @brief Rectangle
 */
struct Rect {
  Rect() : px(0), py(0), width(0), height(0){};
  Rect(int x, int y, int w, int h) : px(x), py(y), width(w), height(h){};

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive&, const unsigned int) {}

  bool operator==(const Rect& rhs) const {
    return (px == rhs.px) && (py == rhs.py) && (width == rhs.width) &&
           (height == rhs.height);
  }

  // The top left point of the rectangle
  int px;
  int py;
  // The width and height of the rectangle
  int width;
  int height;
};

/**
 * @brief Prediction result, a string label and a confidence score
 */
typedef std::pair<std::string, float> Prediction;

class Stream;
typedef std::shared_ptr<Stream> StreamPtr;
class Camera;
typedef std::shared_ptr<Camera> CameraPtr;
class Pipeline;
typedef std::shared_ptr<Pipeline> PipelinePtr;
class Frame;
typedef std::shared_ptr<Frame> FramePtr;
class Processor;
typedef std::shared_ptr<Processor> ProcessorPtr;

typedef std::unordered_map<std::string, std::string> FactoryParamsType;

//// Model types
enum ModelType {
  MODEL_TYPE_INVALID = 0,
  MODEL_TYPE_CAFFE,
  MODEL_TYPE_TENSORFLOW
};

//// Camera types
enum CameraType { CAMERA_TYPE_GST = 0, CAMERA_TYPE_PTGRAY, CAMERA_TYPE_VIMBA };

enum CameraModeType {
  CAMERA_MODE_0 = 0,
  CAMERA_MODE_1,
  CAMERA_MODE_2,
  CAMERA_MODE_3,
  CAMERA_MODE_COUNT,
  CAMERA_MODE_INVALID
};

enum CamereaFeatureType {
  CAMERA_FEATURE_INVALID = 0,
  CAMERA_FEATURE_EXPOSURE,
  CAMERA_FEATURE_GAIN,
  CAMERA_FEATURE_SHUTTER,
  CMAERA_FEATURE_IMAGE_SIZE,
  CAMERA_FEATURE_MODE
};

enum CameraImageSizeType {
  CAMERA_IMAGE_SIZE_INVALID = 0,
  CAMERA_IMAGE_SIZE_800x600,
  CAMERA_IMAGE_SIZE_1600x1200,
  CAMEAR_IMAGE_SIZE_1920x1080,
};

enum CameraPixelFormatType {
  CAMERA_PIXEL_FORMAT_INVALID = 0,
  CAMERA_PIXEL_FORMAT_RAW8,
  CAMERA_PIXEL_FORMAT_RAW12,
  CAMERA_PIXEL_FORMAT_MONO8,
  CAMERA_PIXEL_FORMAT_BGR,
  CAMERA_PIXEL_FORMAT_YUV411,
  CAMERA_PIXEL_FORMAT_YUV422,
  CAMERA_PIXEL_FORMAT_YUV444
};

std::string GetCameraPixelFormatString(CameraPixelFormatType pfmt);

//// Processor types
enum ProcessorType {
  PROCESSOR_TYPE_CAMERA = 0,
  PROCESSOR_TYPE_CUSTOM,
  PROCESSOR_TYPE_ENCODER,
  PROCESSOR_TYPE_FILE_WRITER,
#ifdef USE_RPC
  PROCESSOR_TYPE_FRAME_RECEIVER,
  PROCESSOR_TYPE_FRAME_SENDER,
#endif  // USE_RPC
#ifdef USE_ZMQ
  PROCESSOR_TYPE_FRAME_PUBLISHER,
  PROCESSOR_TYPE_FRAME_SUBSCRIBER,
#endif  // USE_ZMQ
  PROCESSOR_TYPE_IMAGE_CLASSIFIER,
  PROCESSOR_TYPE_IMAGE_SEGMENTER,
  PROCESSOR_TYPE_IMAGE_TRANSFORMER,
  PROCESSOR_TYPE_NEURAL_NET_EVALUATOR,
  PROCESSOR_TYPE_OPENCV_FACE_DETECTOR,
  PROCESSOR_TYPE_THROTTLER,
  PROCESSOR_TYPE_INVALID
};
// Returns the ProcessorType enum value corresponding to the string.
inline ProcessorType GetProcessorTypeByString(const std::string& type) {
  if (type == "Camera") {
    return PROCESSOR_TYPE_CAMERA;
  } else if (type == "Custom") {
    return PROCESSOR_TYPE_CUSTOM;
  } else if (type == "GstVideoEncoder") {
    return PROCESSOR_TYPE_ENCODER;
  } else if (type == "FileWriter") {
    return PROCESSOR_TYPE_FILE_WRITER;
#ifdef USE_RPC
  } else if (type == "FrameReceiver") {
    return PROCESSOR_TYPE_FRAME_RECEIVER;
  } else if (type == "FrameSender") {
    return PROCESSOR_TYPE_FRAME_SENDER;
#endif  // USE_RPC
#ifdef USE_ZMQ
  } else if (type == "FramePublisher") {
    return PROCESSOR_TYPE_FRAME_PUBLISHER;
  } else if (type == "FrameSubscriber") {
    return PROCESSOR_TYPE_FRAME_SUBSCRIBER;
#endif  // USE_ZMQ
  } else if (type == "ImageClassifier") {
    return PROCESSOR_TYPE_IMAGE_CLASSIFIER;
  } else if (type == "ImageSegmenter") {
    return PROCESSOR_TYPE_IMAGE_SEGMENTER;
  } else if (type == "ImageTransformer") {
    return PROCESSOR_TYPE_IMAGE_TRANSFORMER;
  } else if (type == "NeuralNetEvaluator") {
    return PROCESSOR_TYPE_NEURAL_NET_EVALUATOR;
  } else if (type == "OpenCVFaceDetector") {
    return PROCESSOR_TYPE_OPENCV_FACE_DETECTOR;
  } else if (type == "Throttler") {
    return PROCESSOR_TYPE_THROTTLER;
  } else {
    return PROCESSOR_TYPE_INVALID;
  }
}

// Returns a human-readable string corresponding to the provided ProcessorType.
inline std::string GetStringForProcessorType(ProcessorType type) {
  switch (type) {
    case PROCESSOR_TYPE_CAMERA:
      return "Camera";
    case PROCESSOR_TYPE_CUSTOM:
      return "Custom";
    case PROCESSOR_TYPE_ENCODER:
      return "GstVideoEncoder";
    case PROCESSOR_TYPE_FILE_WRITER:
      return "FileWriter";
#ifdef USE_RPC
    case PROCESSOR_TYPE_FRAME_RECEIVER:
      return "FrameReceiver";
    case PROCESSOR_TYPE_FRAME_SENDER:
      return "FrameSender";
#endif  // USE_RPC
#ifdef USE_ZMQ
    case PROCESSOR_TYPE_FRAME_PUBLISHER:
      return "FramePublisher";
    case PROCESSOR_TYPE_FRAME_SUBSCRIBER:
      return "FrameSubscriber";
#endif  // USE_ZMQ
    case PROCESSOR_TYPE_IMAGE_CLASSIFIER:
      return "ImageClassifier";
    case PROCESSOR_TYPE_IMAGE_SEGMENTER:
      return "ImageSegmenter";
    case PROCESSOR_TYPE_IMAGE_TRANSFORMER:
      return "ImageTransformer";
    case PROCESSOR_TYPE_NEURAL_NET_EVALUATOR:
      return "NeuralNetEvaluator";
    case PROCESSOR_TYPE_OPENCV_FACE_DETECTOR:
      return "OpenCVFaceDetector";
    case PROCESSOR_TYPE_THROTTLER:
      return "Throttler";
    case PROCESSOR_TYPE_INVALID:
      return "Invalid";
  }

  LOG(FATAL) << "Unhandled ProcessorType: " << type;
}

#endif  // STREAMER_COMMON_TYPES_H_
