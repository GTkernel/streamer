//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_FRAME_H
#define STREAMER_FRAME_H

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

struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score; /**< Larger score should mean higher confidence. */
};

struct FacePts {
  float x[5],y[5];
};

struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
};

class MetadataFrame : public Frame {
 public:
  MetadataFrame() = delete;
  MetadataFrame(std::vector<string> tags, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<string> tags,
                std::vector<Rect> bboxes,
                cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<FaceInfo> faceInfo, cv::Mat original_image = cv::Mat())
    : Frame(FRAME_TYPE_MD, original_image), faceInfo_(faceInfo) {}
  MetadataFrame(std::vector<FaceInfo> faceInfo,
                std::vector<std::vector<float>> face_features,
                cv::Mat original_image = cv::Mat())
    : Frame(FRAME_TYPE_MD, original_image), faceInfo_(faceInfo), face_features_(face_features) {}
  std::vector<string> GetTags();
  std::vector<Rect> GetBboxes();
  std::vector<FaceInfo> GetFaceInfo() { return faceInfo_; }
  std::vector<std::vector<float>> GetFaceFeatures() { return face_features_; }
  virtual FrameType GetType() override;

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
  std::vector<FaceInfo> faceInfo_;
  std::vector<std::vector<float>> face_features_;
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
