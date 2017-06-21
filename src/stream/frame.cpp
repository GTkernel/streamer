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
  auto it = frame.frame_data_.find("DataBuffer");
  if(it != frame.frame_data_.end()) {
    DataBuffer newbuf(boost::get<DataBuffer>(it->second));
    frame_data_["DataBuffer"] = newbuf;
  }
}

Frame::Frame(const std::unique_ptr<Frame>& frame) : Frame(*frame.get()) {
}

FrameType Frame::GetType() const {
  if(frame_data_.count("DataBuffer") > 0) {
    return FRAME_TYPE_BYTES;
    return FRAME_TYPE_LAYER;
  } else if(frame_data_.count("Tags") > 0 || frame_data_.count("Bboxes") > 0) {
    return FRAME_TYPE_MD;  
  } else if(frame_data_.count("OriginalImage") > 0) {
    return FRAME_TYPE_IMAGE;
  } else {
    return FRAME_TYPE_INVALID;
  }
}

template <typename T>
T Frame::GetValue(std::string key) const {
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

nlohmann::json Frame::ToJson() const {
  nlohmann::json md_j;
  md_j["tags"] = this->GetValue<std::vector<std::string>>("Tags");

  std::vector<nlohmann::json> bboxes;
  for (const auto& bbox : this->GetValue<std::vector<Rect>>("Bboxes")) {
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
template void Frame::SetValue(std::string, const std::vector<Rect>&);
template void Frame::SetValue(std::string, const DataBuffer&);
template void Frame::SetValue(std::string, const cv::Mat&);

template double Frame::GetValue(std::string) const;
template float Frame::GetValue(std::string) const;
template int Frame::GetValue(std::string) const;
template std::string Frame::GetValue(std::string) const;
template std::vector<std::string> Frame::GetValue(std::string) const;
template cv::Mat Frame::GetValue(std::string) const;
template DataBuffer Frame::GetValue(std::string) const;
