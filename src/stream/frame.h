//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef TX1DNN_FRAME_H
#define TX1DNN_FRAME_H

#include "frame.h"
#include "common/common.h"

class Frame {
 public:
  Frame() = delete;
  Frame(FrameType frame_type);
  virtual ~Frame(){};
  FrameType GetType();
 private:
  FrameType frame_type_;
};

class ImageFrame : public Frame {
 public:
  ImageFrame(cv::Mat image, cv::Mat original_image);
  ImageFrame(cv::Mat image);
  Shape GetSize();
  cv::Mat GetImage();
  cv::Mat GetOriginalImage();
  void SetOriginalImage(cv::Mat original_image);
  void SetImage(cv::Mat image);

 private:
  cv::Mat image_;
  cv::Mat original_image_;
  Shape shape_;
};

class MetadataFrame : public Frame {
 public:
  MetadataFrame() = delete;
  MetadataFrame(string tag);
  MetadataFrame(float p1x, float p1y, float p2x, float p2y);
  string GetTag();
  const float *GetBbox();
 private:
  string tag_;
  float bbox_[4];
};

#endif //TX1DNN_FRAME_H
