//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_FRAME_H
#define STREAMER_FRAME_H

#include <bitset>
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

class MetadataFrame : public Frame {
 public:
  enum Bit{
    Bit_tags = 0,
    Bit_bboxes,
    Bit_face_landmarks,
    Bit_face_features,
  };
  MetadataFrame() = delete;
  MetadataFrame(cv::Mat original_image = cv::Mat())
    : Frame(FRAME_TYPE_MD, original_image) {}
  MetadataFrame(std::vector<string> tags, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<string> tags,
                std::vector<Rect> bboxes,
                cv::Mat original_image = cv::Mat());
  MetadataFrame(std::vector<Rect> bboxes,
                std::vector<FaceLandmark> face_landmarks,
                cv::Mat original_image = cv::Mat())
    : Frame(FRAME_TYPE_MD, original_image),
      bboxes_(bboxes),
      face_landmarks_(face_landmarks) {
    bitset_.set(Bit_bboxes);
    bitset_.set(Bit_face_landmarks);
  }
  MetadataFrame(std::vector<Rect> bboxes,
                std::vector<FaceLandmark> face_landmarks,
                std::vector<std::vector<float>> face_features,
                cv::Mat original_image = cv::Mat())
    : Frame(FRAME_TYPE_MD, original_image),
      bboxes_(bboxes),
      face_landmarks_(face_landmarks),
      face_features_(face_features) {
    bitset_.set(Bit_bboxes);
    bitset_.set(Bit_face_landmarks);
    bitset_.set(Bit_face_features);
  }
  std::vector<string> GetTags();
  std::vector<Rect> GetBboxes();
  std::vector<FaceLandmark> GetFaceLandmarks() { return face_landmarks_; }
  std::vector<std::vector<float>> GetFaceFeatures() { return face_features_; }
  std::bitset<32> GetBitset() { return bitset_; }
  virtual FrameType GetType() override;

 private:
  std::vector<string> tags_;
  std::vector<Rect> bboxes_;
  std::vector<FaceLandmark> face_landmarks_;
  std::vector<std::vector<float>> face_features_;
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
