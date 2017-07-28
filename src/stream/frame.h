//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#ifndef STREAMER_STREAM_FRAME_H_
#define STREAMER_STREAM_FRAME_H_

#include "common/types.h"

#include <unordered_map>
#include <unordered_set>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/variant.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <json/src/json.hpp>

#include "common/common.h"
#include "common/context.h"

class Frame {
 public:
  Frame(double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  Frame(const std::unique_ptr<Frame>& frame);
  // Creates a new Frame object that is a copy of "frame".
  Frame(const Frame& frame);
  // Creates a new Frame object that contains the fields in "fields" copied from
  // "frame". If "fields" is empty, then all fields will be copied.
  Frame(const Frame& frame, std::unordered_set<std::string> fields);

  template <typename T>
  void SetValue(std::string key, const T& val);
  template <typename T>
  T GetValue(std::string key) const;
  std::string ToString() const;
  nlohmann::json ToJson() const;
  using field_types =
      boost::variant<int, std::string, float, double, unsigned long,
                     boost::posix_time::ptime, cv::Mat, std::vector<char>,
                     std::vector<std::string>, std::vector<double>,
                     std::vector<Rect>>;
  using map_size_type = std::unordered_map<std::string, field_types>::size_type;
  map_size_type Count(std::string key) const { return frame_data_.count(key); }

 private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& frame_data_;
  }

  std::unordered_map<std::string, field_types> frame_data_;
};

#endif  // STREAMER_STREAM_FRAME_H_
