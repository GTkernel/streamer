//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_FRAME_H
#define STREAMER_FRAME_H

#include <bitset>
#include <boost/optional.hpp>
#include "common/common.h"
#include "frame.h"
#include "cv.h"

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

struct PointFeature {
  PointFeature(const cv::Point& p, const std::vector<float>& f)
    : point(p), face_feature(f) {}
  cv::Point point;
  std::vector<float> face_feature;
};

class MetadataFrame : public Frame {
 public:
  enum Bit{
    Bit_tags = 0,
    Bit_bboxes,
    Bit_face_landmarks,
    Bit_face_features,
    Bit_confidences,
    Bit_paths,
  };
  MetadataFrame() = delete;
  MetadataFrame(cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<string> tags, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat());
  std::vector<string> GetTags();
  void SetTags(const std::vector<string>& tags);
  std::vector<Rect> GetBboxes();
  void SetBboxes(const std::vector<Rect>& bboxes);
  std::vector<FaceLandmark> GetFaceLandmarks();
  void SetFaceLandmarks(const std::vector<FaceLandmark>& face_landmarks);
  std::vector<std::vector<float>> GetFaceFeatures();
  void SetFaceFeatures(const std::vector<std::vector<float>>& face_features);
  std::vector<float> GetConfidences();
  void SetConfidences(const std::vector<float>& confidences);
  std::list<std::list<boost::optional<PointFeature>>> GetPaths();
  void SetPaths(const std::list<std::list<boost::optional<PointFeature>>>& paths);
  std::bitset<32> GetBitset();
  void RenderAll();
  virtual FrameType GetType() override;

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
  std::vector<FaceLandmark> face_landmarks_;
  std::vector<std::vector<float>> face_features_;
  std::vector<float> confidences_;
  std::list<std::list<boost::optional<PointFeature>>> paths_;
  std::bitset<32> bitset_;
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
