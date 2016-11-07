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
  PROCESSOR_TYPE_CUSTOM,

  PROCESSOR_TYPE_CAMERA,
  PROCESSOR_TYPE_ENCODER,
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
  } else if (str == "Encoder") {
    return PROCESSOR_TYPE_ENCODER;
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
