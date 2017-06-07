//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_FRAME_H
#define STREAMER_FRAME_H

#include "json/json.hpp"

#include "common/common.h"
#include "common/context.h"

class Frame {
 public:
  Frame() = delete;
  Frame(FrameType frame_type, cv::Mat original_image,
        double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  virtual ~Frame(){};
  FrameType GetType();
  cv::Mat GetOriginalImage();
  void SetOriginalImage(cv::Mat original_image);
  double GetStartTime();

 private:
  FrameType frame_type_;
  cv::Mat original_image_;
  // Time since streamer context was started
  double start_time_;
};

class ImageFrame : public Frame {
 public:
  ImageFrame(
      cv::Mat image, cv::Mat original_image = cv::Mat(),
      double start_time = Context::GetContext().GetTimer().ElapsedMSec());
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
  MetadataFrame(
      std::vector<string> tags, cv::Mat original_image = cv::Mat(),
      double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  MetadataFrame(
      std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat(),
      double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  MetadataFrame(nlohmann::json j);
  std::vector<string> GetTags() const;
  std::vector<Rect> GetBboxes() const;
  nlohmann::json ToJson() const;

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
};

class BytesFrame : public Frame {
 public:
  BytesFrame() = delete;
  BytesFrame(
      DataBuffer data_buffer, cv::Mat original_image = cv::Mat(),
      double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  DataBuffer GetDataBuffer();

 private:
  DataBuffer data_buffer_;
};

class LayerFrame : public Frame {
 public:
  LayerFrame() = delete;
  LayerFrame(std::string layer_name, cv::Mat activations,
             cv::Mat original_image = cv::Mat());
  const std::string GetLayerName() const;
  cv::Mat GetActivations() const;

 private:
  const std::string layer_name_;
  cv::Mat activations_;
};

#endif  // STREAMER_FRAME_H
