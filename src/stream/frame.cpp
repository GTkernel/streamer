//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"
#include <opencv2/core/core.hpp>

Frame::Frame(FrameType frame_type, cv::Mat original_image)
    : frame_type_(frame_type), original_image_(original_image) {}

cv::Mat Frame::GetOriginalImage() { return original_image_; }

void Frame::SetOriginalImage(cv::Mat original_image) {
  original_image_ = original_image;
}

ImageFrame::ImageFrame(cv::Mat image, cv::Mat original_image)
    : Frame(FRAME_TYPE_IMAGE, original_image),
      image_(image),
      shape_(image.channels(), image.cols, image.rows) {}

Shape ImageFrame::GetSize() { return shape_; }

cv::Mat ImageFrame::GetImage() { return image_; }

void ImageFrame::SetImage(cv::Mat image) { image_ = image; }
FrameType ImageFrame::GetType() { return FRAME_TYPE_IMAGE; }
FrameType Frame::GetType() { return frame_type_; }

MetadataFrame::MetadataFrame(cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image) {}
MetadataFrame::MetadataFrame(std::vector<string> tags, cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image), tags_(tags) {
  bitset_.set(Bit_tags);
}
MetadataFrame::MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image), bboxes_(bboxes) {
  bitset_.set(Bit_bboxes);
}
std::vector<string> MetadataFrame::GetTags() { return tags_; }
void MetadataFrame::SetTags(const std::vector<string>& tags) {
  tags_ = tags;
  bitset_.set(Bit_tags);
}
std::vector<Rect> MetadataFrame::GetBboxes() { return bboxes_; }
void MetadataFrame::SetBboxes(const std::vector<Rect>& bboxes) {
  bboxes_ = bboxes;
  bitset_.set(Bit_bboxes);
}
std::vector<FaceLandmark> MetadataFrame::GetFaceLandmarks() {
  return face_landmarks_;
}
void MetadataFrame::SetFaceLandmarks(const std::vector<FaceLandmark>& face_landmarks) {
  face_landmarks_ = face_landmarks;
  bitset_.set(Bit_face_landmarks);
}
std::vector<std::vector<float>> MetadataFrame::GetFaceFeatures() {
  return face_features_;
}
void MetadataFrame::SetFaceFeatures(const std::vector<std::vector<float>>& face_features) {
  face_features_ = face_features;
  bitset_.set(Bit_face_features);
}
std::vector<float> MetadataFrame::GetConfidences() {
  return confidences_;
}
void MetadataFrame::SetConfidences(const std::vector<float>& confidences) {
  confidences_ = confidences;
  bitset_.set(Bit_confidences);
}
std::list<std::list<boost::optional<PointFeature>>> MetadataFrame::GetPaths() {
  return paths_;
}
void MetadataFrame::SetPaths(const std::list<std::list<boost::optional<PointFeature>>>& paths) {
  paths_ = paths;
  bitset_.set(Bit_paths);
}
std::vector<std::string> MetadataFrame::GetUuids() {
  return uuids_;
}
void MetadataFrame::SetUuids(const std::vector<std::string>& uuids) {
  uuids_ = uuids;
  bitset_.set(Bit_uuids);
}
std::bitset<32> MetadataFrame::GetBitset() {
  return bitset_;
}
FrameType MetadataFrame::GetType() { return FRAME_TYPE_MD; }

BytesFrame::BytesFrame(DataBuffer data_buffer, cv::Mat original_image)
    : Frame(FRAME_TYPE_BYTES, original_image), data_buffer_(data_buffer) {}

DataBuffer BytesFrame::GetDataBuffer() { return data_buffer_; }
FrameType BytesFrame::GetType() { return FRAME_TYPE_BYTES; }
