//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_FRAME_H
#define STREAMER_FRAME_H

#include "common/common.h"
#include "frame.h"

class Frame {
 public:
  Frame() = delete;
  Frame(FrameType frame_type, cv::Mat original_image = cv::Mat());
  virtual ~Frame(){};
  virtual FrameType GetType() = 0;
  cv::Mat GetOriginalImage();
  void SetOriginalImage(cv::Mat original_image);

 private:
  FrameType frame_type_;
  cv::Mat original_image_;
};

class ImageFrame : public Frame {
 public:
  ImageFrame(cv::Mat image, cv::Mat original_image = cv::Mat());
  Shape GetSize();
  cv::Mat GetImage();
  void SetImage(cv::Mat image);

  virtual FrameType GetType() override;

 private:
  cv::Mat image_;
  Shape shape_;
};

class MetadataFrame : public Frame {
 public:
  MetadataFrame() = delete;
  MetadataFrame(std::vector<string> tags, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<string> tags,
                std::vector<Rect> bboxes,
                cv::Mat original_image = cv::Mat());
  std::vector<string> GetTags();
  std::vector<Rect> GetBboxes();
  virtual FrameType GetType() override;

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
};

class BytesFrame : public Frame {
 public:
  BytesFrame() = delete;
  BytesFrame(DataBuffer data_buffer, cv::Mat original_image = cv::Mat());
  DataBuffer GetDataBuffer();
  virtual FrameType GetType() override;

 private:
  DataBuffer data_buffer_;
};

#endif  // STREAMER_FRAME_H
