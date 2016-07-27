//
// Created by Ran Xian on 7/26/16.
//

#ifndef TX1DNN_TYPE_H
#define TX1DNN_TYPE_H

#include <cstdlib>

struct Shape {
  Shape(): channel(0), width(0), height(0){};
  Shape(int c, int w, int h): channel(c), width(w), height(h){};
  Shape(int w, int h): channel(1), width(w), height(h){};
  /**
   * \brief Return volumn (size) of the shape object
   * \return
   */
  size_t Volumn() { return (size_t)channel * width * height; }
  // Number of channels
  int channel;
  // Width
  int width;
  // Height
  int height;
};

#endif //TX1DNN_TYPE_H
