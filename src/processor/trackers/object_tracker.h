// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Multi-target tracking processors
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_TRACKERS_OBJECT_TRACKER_H_
#define STREAMER_PROCESSOR_TRACKERS_OBJECT_TRACKER_H_

#include <cv.h>

#include "processor/processor.h"

class BaseTracker {
 public:
  BaseTracker(const std::string& uuid, const std::string& tag)
      : uuid_(uuid), tag_(tag) {}
  virtual ~BaseTracker() {}
  std::string GetUuid() const { return uuid_; }
  std::string GetTag() const { return tag_; }
  virtual void Initialise(const cv::Mat& gray_image, cv::Rect bb) = 0;
  virtual bool IsInitialised() = 0;
  virtual void Track(const cv::Mat& gray_image) = 0;
  virtual cv::Rect GetBB() = 0;
  virtual std::vector<double> GetBBFeature() = 0;

 private:
  std::string uuid_;
  std::string tag_;
};

class ObjectTracker : public Processor {
 public:
  ObjectTracker(const std::string& type, float calibration_duration);
  static std::shared_ptr<ObjectTracker> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string type_;
  std::list<std::shared_ptr<BaseTracker>> tracker_list_;
  cv::Mat gray_image_;
  float calibration_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_calibration_time_;
};

#endif  // STREAMER_PROCESSOR_TRAKCERS_OBJECT_TRACKER_H_
