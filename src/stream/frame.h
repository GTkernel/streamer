//
//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_STREAM_FRAME_H_
#define STREAMER_STREAM_FRAME_H_

#include "json/json.hpp"
#include "common/types.h"

#include "boost/variant.hpp"
#include "common/common.h"
#include "common/context.h"

#define IMAGE_KEY "Image"
#define BBOXES_KEY "Bboxes"
#define START_TIME_KEY "StartTime"
#define LAYER_NAME_KEY "LayerName"

class Frame {
 public:
  Frame(double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  Frame(const std::unique_ptr<Frame>& frame);
  Frame(const Frame& frame);
  FrameType GetType() const;
  template<typename T>
  void SetValue(std::string key, const T& val);
  template <typename T>
  T GetValue(std::string key) const;
  nlohmann::json ToJson() const;
  using field_types = boost::variant<int, std::string, float, double, cv::Mat, DataBuffer,
                                      std::vector<std::string>, std::vector<Rect>>;

 private:
  std::unordered_map<std::string, field_types> frame_data_;
};
#endif  // STREAMER_STREAM_FRAME_H_
