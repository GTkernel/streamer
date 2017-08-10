//
// Created by Abhinav Garlapati (abhinav2710@gmail.com) on 1/21/17.
//

#ifndef STREAMER_PROCESSOR_OPENCV_PEOPLE_DETECTOR_H_
#define STREAMER_PROCESSOR_OPENCV_PEOPLE_DETECTOR_H_

#ifdef USE_CUDA
#include <opencv2/cudaobjdetect/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif  // USE_CUDA
#include "common/common.h"
#include "processor.h"

class OpenCVPeopleDetector : public Processor {
 public:
  OpenCVPeopleDetector();
  static std::shared_ptr<OpenCVPeopleDetector> Create(
      const FactoryParamsType& params);

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
#endif  // USE_CUDA
};

#endif  // STREAMER_PROCESSOR_OPENCV_PEOPLE_DETECTOR_H_
