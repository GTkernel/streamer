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
  Frame(FrameType frame_type, cv::Mat original_image);
  virtual ~Frame(){};
  FrameType GetType();
  cv::Mat GetOriginalImage();
  void SetOriginalImage(cv::Mat original_image);
 private:
  FrameType frame_type_;
  cv::Mat original_image_;
};

class ImageFrame : public Frame {
 public:
  ImageFrame(cv::Mat image, cv::Mat original_image);
  ImageFrame(cv::Mat image);
  Shape GetSize();
  cv::Mat GetImage();
  void SetImage(cv::Mat image);

 private:
  cv::Mat image_;
  Shape shape_;
};

class MetadataFrame : public Frame {
 public:
  MetadataFrame() = delete;
  MetadataFrame(string tag);
  MetadataFrame(float p1x, float p1y, float p2x, float p2y);
  MetadataFrame(string tag, cv::Mat original_image);
  MetadataFrame(float p1x, float p1y, float p2x, float p2y, cv::Mat original_image);
  string GetTag();
  const float *GetBbox();
 private:
  string tag_;
  float bbox_[4];
};

#endif //TX1DNN_FRAME_H
