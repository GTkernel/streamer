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

// TODO: What's the right solution for this?
#define ORIGINAL_IMAGE_KEY "OriginalImage"
#define DATABUFFER_KEY "DataBuffer"
#define IMAGE_KEY "Image"
#define TAGS_KEY "Tags"
#define BBOXES_KEY "Bboxes"
#define ACTIVATIONS_KEY "Activations"
#define START_TIME_KEY "StartTime"
#define LAYER_NAME_KEY "LayerName"

class Frame {
 public:
  Frame(double start_time = Context::GetContext().GetTimer().ElapsedMSec());
  FrameType GetType() const;
  void SetOriginalImage(cv::Mat original_image);
  cv::Mat GetOriginalImage() const;
  void SetDataBuffer(const DataBuffer& buf);
  DataBuffer GetDataBuffer() const;
  void SetImage(cv::Mat image);
  cv::Mat GetImage() const;
  void SetTags(std::vector<std::string> tags);
  std::vector<std::string> GetTags() const;
  void SetBboxes(std::vector<Rect> bboxes);
  std::vector<Rect> GetBboxes() const;
  void SetActivations(cv::Mat activations);
  cv::Mat GetActivations() const;
  void SetStartTime(double start_time);
  double GetStartTime() const;
  void SetLayerName(std::string layer_name);
  std::string GetLayerName() const;
  nlohmann::json ToJson() const;
  using field_types = boost::variant<int, std::string, float, double, cv::Mat, DataBuffer,
                                      std::vector<std::string>, std::vector<Rect>>;

 private:
  std::unordered_map<std::string, field_types> frame_data_;
};
#endif  // STREAMER_STREAM_FRAME_H_
