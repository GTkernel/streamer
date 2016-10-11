//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef TX1DNN_FRAME_H
#define TX1DNN_FRAME_H

#include "common/common.h"
#include "frame.h"

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
  MetadataFrame(std::vector<string> tags, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat());
  std::vector<string> GetTags();
  std::vector<Rect> GetBboxes();

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
};

#endif  // TX1DNN_FRAME_H
