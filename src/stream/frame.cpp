//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "json/json.hpp"

#include "frame.h"
#include <opencv2/core/core.hpp>

Frame::Frame(FrameType frame_type, cv::Mat original_image)
    : frame_type_(frame_type), original_image_(original_image) {}

cv::Mat Frame::GetOriginalImage() { return original_image_; }

void Frame::SetOriginalImage(cv::Mat original_image) {
  original_image_ = original_image;
}

FrameType Frame::GetType() { return frame_type_; }

ImageFrame::ImageFrame(cv::Mat image, cv::Mat original_image)
    : Frame(FRAME_TYPE_IMAGE, original_image),
      image_(image),
      shape_(image.channels(), image.cols, image.rows) {}

Shape ImageFrame::GetSize() { return shape_; }

cv::Mat ImageFrame::GetImage() { return image_; }

void ImageFrame::SetImage(cv::Mat image) { image_ = image; }

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
MetadataFrame::MetadataFrame(nlohmann::json j)
    : Frame(FRAME_TYPE_MD, cv::Mat()) {
  try {
    nlohmann::json md_j = j.at("MetadataFrame");
    this->tags_ = md_j.at("tags").get<std::vector<std::string>>();

    for (const auto bbox_j :
         md_j.at("bboxes").get<std::vector<nlohmann::json>>()) {
      this->bboxes_.push_back(Rect(bbox_j));
    }
  } catch (std::out_of_range) {
    LOG(FATAL) << "Malformed MetadataFrame JSON: " << j.dump();
  }
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
std::vector<std::vector<double>> MetadataFrame::GetStruckFeatures() {
  return struck_features_;
}
void MetadataFrame::SetStruckFeatures(const std::vector<std::vector<double>>& struck_features) {
  struck_features_ = struck_features;
  bitset_.set(Bit_struck_features);
}
std::bitset<32> MetadataFrame::GetBitset() {
  return bitset_;
}

nlohmann::json MetadataFrame::ToJson() {
  nlohmann::json md_j;
  md_j["tags"] = this->GetTags();

  std::vector<nlohmann::json> bboxes;
  for (auto bbox : this->GetBboxes()) {
    bboxes.push_back(bbox.ToJson());
  }
  md_j["bboxes"] = bboxes;

  nlohmann::json j;
  j["MetadataFrame"] = md_j;
  return j;
}

BytesFrame::BytesFrame(DataBuffer data_buffer, cv::Mat original_image)
    : Frame(FRAME_TYPE_BYTES, original_image), data_buffer_(data_buffer) {}

DataBuffer BytesFrame::GetDataBuffer() { return data_buffer_; }
