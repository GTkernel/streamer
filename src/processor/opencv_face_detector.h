//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#ifndef STREAMER_PROCESSOR_OPENCV_FACE_DETECTOR_H_
#define STREAMER_PROCESSOR_OPENCV_FACE_DETECTOR_H_

#ifdef USE_CUDA
#include <opencv2/cudaobjdetect/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif  // USE_CUDA
#include "common/common.h"
#include "common/types.h"
#include "processor/processor.h"

class OpenCVFaceDetector : public Processor {
 public:
  // TODO: Use a configurable path
  OpenCVFaceDetector(
      float idle_duration,
      string classifier_xml_path =
          "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");

  static std::shared_ptr<OpenCVFaceDetector> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  float idle_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_detect_time_;
  string classifier_xml_path_;
#ifdef USE_CUDA
  cv::cuda::CascadeClassifier classifier_;
#else
  cv::CascadeClassifier classifier_;
#endif  // USE_CUDA
};

#endif  // STREAMER_PROCESSOR_OPENCV_FACE_DETECTOR_H_
