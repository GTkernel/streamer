//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef TX1DNN_FRAME_H
#define TX1DNN_FRAME_H

#include "frame.h"
#include "common/common.h"

class Frame {
 public:
  Frame(cv::Mat image, cv::Mat original_image);
  Frame(cv::Mat image);
  Shape GetSize();
  cv::Mat GetImage();
  cv::Mat GetOriginalFrame();
 private:
  cv::Mat image_;
  cv::Mat original_image_;
  Shape shape_;
};

#endif //TX1DNN_FRAME_H
