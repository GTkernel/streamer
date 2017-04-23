#ifndef STREAMER_STRUCK_TRACKER_H
#define STREAMER_STRUCK_TRACKER_H

#include "cv.h"
#include "common/common.h"
#include "processor.h"
#include "struck/src/Tracker.h"
#include "struck/src/Config.h"

class StruckTracker : public Processor {
public:
  StruckTracker();
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  std::list<std::shared_ptr<struck::Tracker>> tracker_list_;
  cv::Mat gray_image_;
  struck::Config conf_;
};

#endif // STREAMER_STRUCK_TRACKER_H