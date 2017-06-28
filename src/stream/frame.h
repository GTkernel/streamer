//
//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_STREAM_FRAME_H_
#define STREAMER_STREAM_FRAME_H_

#include "common/types.h"

#include <boost/variant.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>

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
  template<typename T>
  void SetValue(std::string key, const T& val);
  template <typename T>
  T GetValue(std::string key) const;
  std::string ToString() const;
  using field_types = boost::variant<int, std::string, float, double, cv::Mat, DataBuffer,
                                      std::vector<std::string>, std::vector<Rect>>;

 private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & frame_data_;
  }

  std::unordered_map<std::string, field_types> frame_data_;
};
#endif  // STREAMER_STREAM_FRAME_H_
