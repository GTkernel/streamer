//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"

Frame::Frame(cv::Mat image, cv::Mat original_image)
    : image_(image),
      original_image_(original_image),
      shape_(image.channels(), image.cols, image.rows) {}

Frame::Frame(cv::Mat image)
    : image_(image), shape_(image.channels(), image.cols, image.rows) {}

Shape Frame::GetSize() { return shape_; }

cv::Mat Frame::GetImage() { return image_; }
cv::Mat Frame::GetOriginalFrame() { return original_image_; }
