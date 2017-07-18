//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#ifndef STREAMER_OPENCV_FACE_DETECTOR_H
#define STREAMER_OPENCV_FACE_DETECTOR_H

using namespace std;

#ifdef USE_CUDA
#include <opencv2/cudaobjdetect/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif
#include "common/common.h"
#include "processor.h"

class OpenCVFaceDetector : public Processor {
 public:
  // TODO: Use a configurable path
  OpenCVFaceDetector(
      string classifier_xml_path =
          "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string classifier_xml_path_;
#ifdef USE_CUDA
  cv::cuda::CascadeClassifier classifier_;
#else
  cv::CascadeClassifier classifier_;
#endif
};

#endif  // STREAMER_OPENCV_FACE_DETECTOR_H
