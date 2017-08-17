/**
 * Multi-face tracking using face feature
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_FACE_TRACKER_H_
#define STREAMER_PROCESSOR_FACE_TRACKER_H_

#include <boost/optional.hpp>
#include "common/common.h"
#include "cv.h"
#include "processor.h"

class FaceTracker : public Processor {
 public:
  FaceTracker(size_t rem_size = 5);
  static std::shared_ptr<FaceTracker> Create(const FactoryParamsType& params);

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

#endif  // STREAMER_PROCESSOR_FACE_TRACKER_H_