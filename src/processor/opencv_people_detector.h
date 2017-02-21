//
// Created by Abhinav Garlapati (abhinav2710@gmail.com) on 1/21/17.
//

#ifndef STREAMER_OPENCV_PEOPLE_DETECTOR_H
#define STREAMER_OPENCV_PEOPLE_DETECTOR_H

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "common/common.h"
#include "processor.h"

class OpenCVPeopleDetector : public Processor {

  public:
    OpenCVPeopleDetector();
    virtual ProcessorType GetType() override;

  protected:
    virtual bool Init() override;
    virtual bool OnStop() override;
    virtual void Process() override;

  private:

#ifdef USE_CUDA
  cv::gpu::HOGDescriptor hog_;
#else
    cv::HOGDescriptor hog_;
#endif

};

#endif // STREAMER_OPENCV_PEOPLE_DETECTOR_H
