/**
 * Multi-target tracking using Struck.
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_TRACKERS_STRUCK_TRACKER_H_
#define STREAMER_PROCESSOR_TRACKERS_STRUCK_TRACKER_H_

#include <struck/src/Config.h>
#include <struck/src/Tracker.h>

#include "processor/trackers/object_tracker.h"

static const string STRUCK_CONF_FILENAME = "struck_config.txt";

class StruckTracker : public BaseTracker {
 public:
  StruckTracker(const std::string& uuid, const std::string& tag)
      : BaseTracker(uuid, tag),
        conf_(Context::GetContext().GetConfigFile(STRUCK_CONF_FILENAME)) {
    impl_.reset(new struck::Tracker(conf_));
  }
  virtual ~StruckTracker() {}
  virtual void Initialise(const cv::Mat& gray_image, cv::Rect bb) {
    struck::FloatRect initBB = struck::IntRect(bb.x, bb.y, bb.width, bb.height);
    impl_->Initialise(gray_image, initBB);
  }
  virtual bool IsInitialised() { return impl_->IsInitialised(); }
  virtual void Track(const cv::Mat& gray_image) { impl_->Track(gray_image); }
  virtual cv::Rect GetBB() {
    struck::IntRect r(impl_->GetBB());
    return cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height());
  }
  virtual std::vector<double> GetBBFeature() { return impl_->GetBBFeature(); }

 private:
  std::unique_ptr<struck::Tracker> impl_;
  struck::Config conf_;
};

#endif  // STREAMER_PROCESSOR_STRUCK_TRACKER_H_
