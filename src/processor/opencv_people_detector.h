//
// Created by Abhinav Garlapati (abhinav2710@gmail.com) on 1/21/17.
//

#ifndef STREAMER_OPENCV_PEOPLE_DETECTOR_H
#define STREAMER_OPENCV_PEOPLE_DETECTOR_H

#ifdef USE_CUDA
#include <opencv2/cudaobjdetect/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif
#include "common/common.h"
#include "processor.h"

class OpenCVPeopleDetector : public Processor {
 public:
  OpenCVPeopleDetector();

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
#ifdef USE_CUDA
  // cv::cuda::HOG hog_;
  cv::cuda::HOG hog_;
#else
  cv::HOGDescriptor hog_;
#endif
};

#endif  // STREAMER_OPENCV_PEOPLE_DETECTOR_H
