//
// Created by Ran Xian on 7/26/16.
//

#ifndef TX1DNN_TYPE_H
#define TX1DNN_TYPE_H

#include <cstdlib>
#include <string>

#ifdef USE_FP16
#include <driver_types.h>
#include <cuda_fp16.h>
#endif

/**
 * @brief 3-D shape structure
 */
struct Shape {
  Shape() : channel(0), width(0), height(0) {};
  Shape(int c, int w, int h) : channel(c), width(w), height(h) {};
  Shape(int w, int h) : channel(1), width(w), height(h) {};
  /**
   * @brief Return volumn (size) of the shape object
   */
  size_t GetVolume() const { return (size_t) channel * width * height; }
  // Number of channels
  int channel;
  // Width
  int width;
  // Height
  int height;
};

/**
 * @brief Prediction result, a string label and a confidence score
 */
typedef std::pair<std::string, float> Prediction;

#ifdef USE_FP16
half Cpu_Float2Half(float f);
float Cpu_Half2Float(half h);

/**
 * @brief Float16 type
 */
struct float16 {
  inline float16() { data.x = 0; }

  inline float16(const float &rhs) {
    data = Cpu_Float2Half(rhs);
  }

  inline operator float() const {
    return Cpu_Half2Float(data);
  }

  half data;
};
#endif

#endif //TX1DNN_TYPE_H
