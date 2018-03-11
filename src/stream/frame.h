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

#include "common/context.h"

#include "tensorflow/core/framework/tensor.h"

// Forward declaration to break the cycle:
//   frame.h -> flow_control_entrance.h -> processor.h -> stream.h -> frame.h
class FlowControlEntrance;

class Frame {
 public:
  Frame(double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  Frame(const std::unique_ptr<Frame>& frame);
  // Creates a new Frame object that is a copy of "frame".
  Frame(const Frame& frame);
  // Creates a new Frame object that contains the fields in "fields" copied from
  // "frame". If "fields" is empty, then all fields will be copied.
  Frame(const Frame& frame, std::unordered_set<std::string> fields);

  void SetFlowControlEntrance(FlowControlEntrance* flow_control_entrance);
  FlowControlEntrance* GetFlowControlEntrance();

  template <typename T>
  void SetValue(std::string key, const T& val);
  template <typename T>
  T GetValue(std::string key) const;
  // Deletes the specified key from the frame, if it exists, otherwise does
  // nothing if the key does not exist.
  void Delete(std::string key);
  std::string ToString() const;
  nlohmann::json ToJson() const;
  nlohmann::json GetFieldJson(const std::string& field) const;
  using field_types =
      boost::variant<std::string, float, double, long, unsigned long, bool,
					 //int, std::string, float, double, long, unsigned long, bool,
                     boost::posix_time::ptime, boost::posix_time::time_duration,
                     cv::Mat, tensorflow::Tensor, std::vector<char>, std::vector<std::string>,
                     //cv::Mat, std::vector<char>, std::vector<std::string>,
                     std::vector<double>, std::vector<Rect>,
                     std::vector<FaceLandmark>, std::vector<std::vector<float>>,
                     std::vector<float>, std::vector<std::vector<double>>,
                     std::vector<Frame>, std::vector<int>>;
  size_t Count(std::string key) const;
  std::unordered_map<std::string, field_types> GetFields();
  void SetStopFrame(bool stop_frame);
  bool IsStopFrame() const;
  // Returns the size in bytes of the data contained in the specified fields.
  // Provide the empty set to specify all fields.
  unsigned long GetRawSizeBytes(
      std::unordered_set<std::string> fields = {}) const;

 private:
  friend class boost::serialization::access;

  // If this is not null, then this frame owns a flow control token from the
  // specified FlowControlEntrance. The token should be released when this frame
  // leaves the pipeline or encounters a FlowControlExit processor.
  FlowControlEntrance* flow_control_entrance_ = nullptr;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& frame_data_;
  }

  std::unordered_map<std::string, field_types> frame_data_;
};

#endif  // STREAMER_STREAM_FRAME_H_
