//
// Created by Ran Xian on 7/26/16.
//

#ifndef STREAMER_TYPE_H
#define STREAMER_TYPE_H

#include <cstdlib>
#include <memory>
#include <string>

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

//// Model types
enum ModelType {
  MODEL_TYPE_INVALID = 0,
  MODEL_TYPE_CAFFE,
  MODEL_TYPE_MXNET,
  MODEL_TYPE_GIE,
  MODEL_TYPE_TENSORFLOW
};

//// Camera types
enum CameraType { CAMERA_TYPE_GST = 0, CAMERA_TYPE_PTGRAY };

//// Frame types
enum FrameType {
  FRAME_TYPE_INVALID = 0,
  FRAME_TYPE_IMAGE,
  FRAME_TYPE_MD,
  FRAME_TYPE_BYTES
};

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
