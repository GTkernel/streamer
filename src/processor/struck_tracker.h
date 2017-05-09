/**
* Multi-target tracking using Struck
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_STRUCK_TRACKER_H
#define STREAMER_STRUCK_TRACKER_H

#include "cv.h"
#include "common/common.h"
#include "processor.h"
#include "struck/src/Tracker.h"
#include "struck/src/Config.h"

class Tracker1 : public struck::Tracker {
public:
  Tracker1(const struck::Config& conf, const std::string& uuid, const std::string& tag):
      struck::Tracker(conf), uuid_(uuid), tag_(tag) {}
  ~Tracker1() {}
  std::string GetUuid() const { return uuid_; }
  std::string GetTag() const { return tag_; }

private:
  std::string uuid_;
  std::string tag_;
};

class StruckTracker : public Processor {
public:
  StruckTracker(float calibration_duration);
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  std::list<std::shared_ptr<Tracker1>> tracker_list_;
  cv::Mat gray_image_;
  struck::Config conf_;
  float calibration_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_calibration_time_;
};

#endif // STREAMER_STRUCK_TRACKER_H
