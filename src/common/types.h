//
// Created by Ran Xian on 7/26/16.
//

#ifndef STREAMER_TYPE_H
#define STREAMER_TYPE_H

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

#ifdef USE_FP16
// THIS ORDER MATTERS!
#include <driver_types.h>

#include <cuda_fp16.h>
#endif

#include "json/json.hpp"

#include "common/common.h"

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
  Rect(int x, int y, int w, int h) : px(x), py(y), width(w), height(h){};

  Rect(nlohmann::json j) {
    try {
      nlohmann::json rect_j = j.at("Rect");
      px = rect_j.at("px").get<int>();
      py = rect_j.at("py").get<int>();
      width = rect_j.at("width").get<int>();
      height = rect_j.at("height").get<int>();
    } catch (std::out_of_range) {
      LOG(FATAL) << "Malformed Rect JSON: " << j.dump();
    }
  }

  nlohmann::json ToJson() {
    nlohmann::json rect_j;
    rect_j["px"] = px;
    rect_j["py"] = py;
    rect_j["width"] = width;
    rect_j["height"] = height;
    nlohmann::json j;
    j["Rect"] = rect_j;
    return j;
  }

  bool operator==(const Rect &rhs) const {
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
  MODEL_TYPE_MXNET,
  MODEL_TYPE_GIE,
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

//// Frame types
enum FrameType {
  FRAME_TYPE_INVALID = 0,
  FRAME_TYPE_IMAGE,
  FRAME_TYPE_MD,
  FRAME_TYPE_BYTES
};

//// Processor types
enum ProcessorType {
  PROCESSOR_TYPE_INVALID = 0,
  PROCESSOR_TYPE_IMAGE_CLASSIFIER,
  PROCESSOR_TYPE_IMAGE_TRANSFORMER,
  PROCESSOR_TYPE_OPENCV_FACE_DETECTOR,
  PROCESSOR_TYPE_DUMMY_NN,
  PROCESSOR_TYPE_IMAGE_SEGMENTER,
  PROCESSOR_TYPE_STREAM_PUBLISHER,
  PROCESSOR_TYPE_FILE_WRITER,
  PROCESSOR_TYPE_CUSTOM,

  PROCESSOR_TYPE_CAMERA,
  PROCESSOR_TYPE_ENCODER,
  PROCESSOR_TYPE_DECODER,

  PROCESSOR_TYPE_FRAME_SENDER,
  PROCESSOR_TYPE_FRAME_RECEIVER
};

inline ProcessorType GetProcessorTypeByString(const std::string &str) {
  if (str == "ImageClassifier") {
    return PROCESSOR_TYPE_IMAGE_CLASSIFIER;
  } else if (str == "ImageTransformer") {
    return PROCESSOR_TYPE_IMAGE_TRANSFORMER;
  } else if (str == "OpenCVFaceDetector") {
    return PROCESSOR_TYPE_OPENCV_FACE_DETECTOR;
  } else if (str == "DummyNN") {
    return PROCESSOR_TYPE_DUMMY_NN;
  } else if (str == "Custom") {
    return PROCESSOR_TYPE_CUSTOM;
  } else if (str == "Camera") {
    return PROCESSOR_TYPE_CAMERA;
  } else if (str == "Encoder" || str == "VideoEncoder") {
    return PROCESSOR_TYPE_ENCODER;
  } else if (str == "FileWriter") {
    return PROCESSOR_TYPE_FILE_WRITER;
  } else if (str == "FrameSender") {
    return PROCESSOR_TYPE_FRAME_SENDER;
  } else if (str == "FrameReceiver") {
    return PROCESSOR_TYPE_FRAME_RECEIVER;
  } else {
    return PROCESSOR_TYPE_INVALID;
  }
}

#ifdef USE_FP16
half Cpu_Float2Half(float f);
float Cpu_Half2Float(half h);

/**
 * @brief Float16 type
 */
struct float16 {
  inline float16() { data.x = 0; }

  inline float16(const float &rhs) { data = Cpu_Float2Half(rhs); }

  inline operator float() const { return Cpu_Half2Float(data); }

  half data;
};
#endif

#endif  // STREAMER_TYPE_H
