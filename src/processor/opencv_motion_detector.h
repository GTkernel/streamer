/**
 * Motion detector using opencv
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_OPENCV_MOTION_DETECTOR_H
#define STREAMER_OPENCV_MOTION_DETECTOR_H

#include <chrono>
#include <opencv2/opencv.hpp>
#include "common/common.h"
#include "processor.h"

class OpenCVMotionDetector : public Processor {
 public:
  OpenCVMotionDetector(float threshold = 0.5, float max_duration = 1.0);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  int GetPixels(cv::Mat& image);

 private:
  std::unique_ptr<cv::BackgroundSubtractorMOG2> mog2_;
  bool first_frame_;
  cv::Mat previous_fore_;
  int previous_pixels_;
  std::chrono::time_point<std::chrono::system_clock> last_send_time_;
  float threshold_;
  float max_duration_;
};

#endif  // STREAMER_OPENCV_MOTION_DETECTOR_H
