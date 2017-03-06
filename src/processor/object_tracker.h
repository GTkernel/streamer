#ifndef STREAMER_OBJECT_TRACKER_H
#define STREAMER_OBJECT_TRACKER_H

#include "cv.h"
#include "common/common.h"
#include "processor.h"

class ObjectTracker : public Processor {
public:
  ObjectTracker(size_t rem_size = 5);
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  cv::Point FindPreviousNearest(const cv::Point& point, const std::vector<cv::Point>& points);
  int GetDistance(const cv::Point& a, const cv::Point& b);
  
private:
  std::list<std::map<std::string, std::vector<cv::Point>>> rem_list_;
  size_t rem_size_;
  bool first_frame_;
};

#endif // STREAMER_OBJECT_TRACKER_H