/**
 * Multi-target detection using fathom stick
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_
#define STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_

#include <caffe/caffe.hpp>
#include <set>
#include "common/common.h"
#include "model/model.h"
#include "ncs/ncs.h"
#include "processor.h"

class NcsYoloDetector : public Processor {
 public:
  NcsYoloDetector(
      const ModelDesc& model_desc, Shape input_shape,
      float confidence_threshold, float idle_duration = 0.f,
      const std::set<std::string>& targets = std::set<std::string>());
  static std::shared_ptr<NcsYoloDetector> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  ModelDesc model_desc_;
  Shape input_shape_;
  std::unique_ptr<NCSManager> detector_;
  float confidence_threshold_;
  float idle_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_detect_time_;
  std::set<std::string> targets_;
  std::vector<std::string> voc_names_;
};

#endif  // STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_
