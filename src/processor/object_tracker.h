#ifndef STREAMER_OBJECT_TRACKER_H
#define STREAMER_OBJECT_TRACKER_H

#include <boost/optional.hpp>
#include "cv.h"
#include "common/common.h"
#include "processor.h"

struct PointFeature {
  PointFeature(const cv::Point& p, const std::vector<float>& f)
    : point(p), face_feature(f) {}
  cv::Point point;
  std::vector<float> face_feature;
};

class ObjectTracker : public Processor {
public:
  ObjectTracker(size_t rem_size = 5);
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  boost::optional<PointFeature> FindPreviousNearest(const PointFeature& point_feature,
                                                    std::vector<PointFeature> point_features,
                                                    float threshold);
  float GetDistance(const std::vector<float>& a, const std::vector<float>& b);
  
private:
  std::list<std::vector<PointFeature>> rem_list_;
  size_t rem_size_;
  bool first_frame_;
};

#endif // STREAMER_OBJECT_TRACKER_H