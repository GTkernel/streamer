/**
* Multi-target tracking using Struck etc.
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_OBJ_TRACKER_H
#define STREAMER_OBJ_TRACKER_H

#include "cv.h"
#include "common/common.h"
#include "processor.h"

class BaseTracker {
public:
  BaseTracker(const std::string& uuid, const std::string& tag):
      uuid_(uuid), tag_(tag) {}
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

class ObjTracker : public Processor {
public:
  ObjTracker(const std::string& type, float calibration_duration);
  virtual ProcessorType GetType() override;

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

#endif // STREAMER_OBJ_TRACKER_H