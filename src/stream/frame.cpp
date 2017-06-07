//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//
#include <opencv2/core/core.hpp>
#include "json/json.hpp"

#include "frame.h"

Frame::Frame(FrameType frame_type, cv::Mat original_image, double start_time)
    : frame_type_(frame_type),
      original_image_(original_image),
      start_time_(start_time) {}

cv::Mat Frame::GetOriginalImage() { return original_image_; }

void Frame::SetOriginalImage(cv::Mat original_image) {
  original_image_ = original_image;
}

FrameType Frame::GetType() { return frame_type_; }

double Frame::GetStartTime() { return start_time_; }

ImageFrame::ImageFrame(cv::Mat image, cv::Mat original_image, double start_time)
    : Frame(FRAME_TYPE_IMAGE, original_image, start_time),
      image_(image),
      shape_(image.channels(), image.cols, image.rows) {}

Shape ImageFrame::GetSize() { return shape_; }

cv::Mat ImageFrame::GetImage() { return image_; }

void ImageFrame::SetImage(cv::Mat image) { image_ = image; }

MetadataFrame::MetadataFrame(std::vector<string> tags, cv::Mat original_image,
                             double start_time)
    : Frame(FRAME_TYPE_MD, original_image, start_time), tags_(tags) {}
MetadataFrame::MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image,
                             double start_time)
    : Frame(FRAME_TYPE_MD, original_image, start_time), bboxes_(bboxes) {}

MetadataFrame::MetadataFrame(nlohmann::json j)
    : Frame(FRAME_TYPE_MD, cv::Mat()) {
  try {
    nlohmann::json md_j = j.at("MetadataFrame");
    this->tags_ = md_j.at("tags").get<std::vector<std::string>>();

    for (const auto& bbox_j :
         md_j.at("bboxes").get<std::vector<nlohmann::json>>()) {
      this->bboxes_.push_back(Rect(bbox_j));
    }
  } catch (std::out_of_range) {
    LOG(FATAL) << "Malformed MetadataFrame JSON: " << j.dump();
  }
}

std::vector<string> MetadataFrame::GetTags() const { return tags_; }
std::vector<Rect> MetadataFrame::GetBboxes() const { return bboxes_; }

nlohmann::json MetadataFrame::ToJson() const {
  nlohmann::json md_j;
  md_j["tags"] = this->GetTags();

  std::vector<nlohmann::json> bboxes;
  for (const auto& bbox : this->GetBboxes()) {
    bboxes.push_back(bbox.ToJson());
  }
  md_j["bboxes"] = bboxes;

  nlohmann::json j;
  j["MetadataFrame"] = md_j;
  return j;
}

BytesFrame::BytesFrame(DataBuffer data_buffer, cv::Mat original_image,
                       double start_time)
    : Frame(FRAME_TYPE_BYTES, original_image, start_time),
      data_buffer_(data_buffer) {}

DataBuffer BytesFrame::GetDataBuffer() { return data_buffer_; }

LayerFrame::LayerFrame(std::string layer_name, cv::Mat activations,
                       cv::Mat original_image)
    : Frame(FRAME_TYPE_LAYER, original_image),
      layer_name_(layer_name),
      activations_(activations) {}

const std::string LayerFrame::GetLayerName() const { return layer_name_; }

cv::Mat LayerFrame::GetActivations() const { return activations_; }
