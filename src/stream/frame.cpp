//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"
#include <opencv2/core/core.hpp>

Frame::Frame(cv::Mat image, cv::Mat original_image)
    : image_(image),
      original_image_(original_image),
      shape_(image.channels(), image.cols, image.rows) {}

Frame::Frame(cv::Mat image)
    : image_(image), shape_(image.channels(), image.cols, image.rows) {}

Shape Frame::GetSize() { return shape_; }

cv::Mat Frame::GetImage() { return image_; }
cv::Mat Frame::GetOriginalImage() { return original_image_; }

void Frame::SetOriginalImage(cv::Mat original_image) {
  original_image_ = original_image;
}

void Frame::SetImage(cv::Mat image) {
  image_ = image;
}
