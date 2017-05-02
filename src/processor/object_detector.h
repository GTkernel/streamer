/**
* Multi-target detection using FRCNN
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_OBJECT_DETECTOR_H
#define STREAMER_OBJECT_DETECTOR_H

#include <set>
#include "common/common.h"
#include "model/model.h"
#include "processor.h"
#include "api/api.hpp"

class ObjectDetector : public Processor {
public:
  ObjectDetector(const ModelDesc &model_desc,
                 Shape input_shape,
                 float idle_duration = 0.f,
                 const std::set<std::string>& targets = std::set<std::string>());
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  ModelDesc model_desc_;
  Shape input_shape_;
  std::unique_ptr<API::Detector> detector_;
  float idle_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_detect_time_;
  std::set<std::string> targets_;
};

#endif // STREAMER_OBJECT_DETECTOR_H
