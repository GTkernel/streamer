/**
 * Multi-target detection using FRCNN
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_OBJECT_DETECTOR_H_
#define STREAMER_PROCESSOR_OBJECT_DETECTOR_H_

#include <set>
#include "common/common.h"
#include "model/model.h"
#include "processor.h"

struct ObjectInfo {
  ObjectInfo() { face_landmark_flag = false; }
  std::string tag;
  cv::Rect bbox;
  float confidence;
  FaceLandmark face_landmark;
  bool face_landmark_flag;
};

class BaseDetector {
 public:
  BaseDetector() {}
  virtual ~BaseDetector() {}
  virtual bool Init() = 0;
  virtual std::vector<ObjectInfo> Detect(const cv::Mat& image) = 0;
};

class ObjectDetector : public Processor {
 public:
  ObjectDetector(
      const std::string& type, const std::vector<ModelDesc>& model_descs,
      Shape input_shape, float confidence_threshold, float idle_duration = 0.f,
      const std::set<std::string>& targets = std::set<std::string>());
  static std::shared_ptr<ObjectDetector> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string type_;
  std::vector<ModelDesc> model_descs_;
  Shape input_shape_;
  float confidence_threshold_;
  float idle_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_detect_time_;
  std::set<std::string> targets_;
  std::unique_ptr<BaseDetector> detector_;
};

#endif  // STREAMER_PROCESSOR_OBJECT_DETECTOR_H_
