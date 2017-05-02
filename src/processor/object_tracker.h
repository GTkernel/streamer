/**
* Multi-face tracking using face feature
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_OBJECT_TRACKER_H
#define STREAMER_OBJECT_TRACKER_H

#include <boost/optional.hpp>
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
  void AttachNearest(std::vector<PointFeature>& point_features,
                     float threshold);
  float GetDistance(const std::vector<float>& a, const std::vector<float>& b);
  
private:
  std::list<std::list<boost::optional<PointFeature>>> path_list_;
  size_t rem_size_;
  bool first_frame_;
};

#endif // STREAMER_OBJECT_TRACKER_H