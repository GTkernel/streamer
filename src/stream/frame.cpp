//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//
#include "frame.h"

#include <opencv2/core/core.hpp>

#include "json/json.hpp"
#include "common/types.h"

Frame::Frame(double start_time) {
  frame_data_[START_TIME_KEY] = start_time;
}

Frame::Frame(const Frame& frame) {
  frame_data_ = frame.frame_data_;
  // Deep copy the databuffer
  auto it = frame.frame_data_.find(DATABUFFER_KEY);
  if(it != frame.frame_data_.end()) {
    DataBuffer newbuf(boost::get<DataBuffer>(it->second));
    frame_data_[DATABUFFER_KEY] = newbuf;
  }
}

Frame::Frame(const std::unique_ptr<Frame>& frame) : Frame(*frame.get()) {
}

FrameType Frame::GetType() const {
  if(frame_data_.count(DATABUFFER_KEY) > 0) {
    return FRAME_TYPE_BYTES;
  } else if(frame_data_.count(ACTIVATIONS_KEY) > 0) {
    return FRAME_TYPE_LAYER;
  } else if(frame_data_.count(TAGS_KEY) > 0 || frame_data_.count(BBOXES_KEY) > 0) {
    return FRAME_TYPE_MD;  
  } else if(frame_data_.count(ORIGINAL_IMAGE_KEY) > 0) {
    return FRAME_TYPE_IMAGE;
  } else {
    return FRAME_TYPE_INVALID;
  }
}

template <typename T>
T Frame::GetValue(std::string key) {
  auto it = frame_data_.find(key);
  if (it != frame_data_.end()) {
    return boost::get<T>(it->second);
  } else {
    throw std::runtime_error("No key " + key + " in Frame\n");
  }
}

template <typename T>
void Frame::SetValue(std::string key, const T& val) {
  frame_data_[key] = val;
}

void Frame::SetOriginalImage(cv::Mat original_image) {
  frame_data_[ORIGINAL_IMAGE_KEY] = original_image;
}

cv::Mat Frame::GetOriginalImage() const {
  auto it = frame_data_.find(ORIGINAL_IMAGE_KEY);
  if(it != frame_data_.end()) {
    return boost::get<cv::Mat>(it->second);
  } else {
    throw std::runtime_error("No original image in Frame\n");
  }
}

void Frame::SetDataBuffer(const DataBuffer& buf) {
  frame_data_[DATABUFFER_KEY] = buf;
}

// Databuffer is often used in conjunction with pointer aliasing
// so const really doesn't mean that much here.
DataBuffer Frame::GetDataBuffer() const {
  auto it = frame_data_.find(DATABUFFER_KEY);
  if(it != frame_data_.end()) {
    return boost::get<DataBuffer>(it->second);
  } else {
    throw std::runtime_error("No raw image in Frame\n");
  }
}

void Frame::SetImage(cv::Mat image) {
  frame_data_[IMAGE_KEY] = image;
}

cv::Mat Frame::GetImage() const {
  auto it = frame_data_.find(IMAGE_KEY);
  if(it != frame_data_.end()) {
    return boost::get<cv::Mat>(it->second);
  } else {
    throw std::runtime_error("No image in Frame\n");
  }
}

void Frame::SetTags(std::vector<std::string> tags) {
  frame_data_[TAGS_KEY] = tags;
}

std::vector<std::string> Frame::GetTags() const {
  auto it = frame_data_.find(TAGS_KEY);
  if(it != frame_data_.end()) {
    return boost::get<std::vector<std::string>>(it->second);
  } else {
    return {};
  }
}

void Frame::SetBboxes(std::vector<Rect> bboxes) {
  frame_data_[BBOXES_KEY] = bboxes;
}

std::vector<Rect> Frame::GetBboxes() const {
  auto it = frame_data_.find(BBOXES_KEY);
  if(it != frame_data_.end()) {
    return boost::get<std::vector<Rect>>(it->second);
  } else {
    return {};
  }
}

void Frame::SetActivations(cv::Mat activations) {
  frame_data_[ACTIVATIONS_KEY] = activations;
}

cv::Mat Frame::GetActivations() const {
  auto it = frame_data_.find(ACTIVATIONS_KEY);
  if(it != frame_data_.end()) {
    return boost::get<cv::Mat>(it->second);
  } else {
    throw std::runtime_error("No activations in Frame\n");
  }
}

void Frame::SetStartTime(double start_time) {
  frame_data_[START_TIME_KEY] = start_time;
}

double Frame::GetStartTime() const {
  auto it = frame_data_.find(START_TIME_KEY);
  if(it != frame_data_.end()) {
    return boost::get<double>(it->second);
  } else {
    throw std::runtime_error("No start time in Frame\n");
  }
}

void Frame::SetLayerName(std::string layer_name) {
  frame_data_[LAYER_NAME_KEY] = layer_name;
}

std::string Frame::GetLayerName() const {
  auto it = frame_data_.find(LAYER_NAME_KEY);
  if(it != frame_data_.end()) {
    return boost::get<std::string>(it->second);
  } else {
    throw std::runtime_error("No start time in Frame\n");
  }
}

nlohmann::json Frame::ToJson() const {
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

// Types declared in field_types of frame
template void Frame::SetValue(std::string, const double&);
template void Frame::SetValue(std::string, const float&);
template void Frame::SetValue(std::string, const int&);
template void Frame::SetValue(std::string, const std::string&);
template void Frame::SetValue(std::string, const std::vector<std::string>&);
template void Frame::SetValue(std::string, const cv::Mat&);

template double Frame::GetValue(std::string);
template float Frame::GetValue(std::string);
template int Frame::GetValue(std::string);
template std::string Frame::GetValue(std::string);
template std::vector<std::string> Frame::GetValue(std::string);
template cv::Mat Frame::GetValue(std::string);
