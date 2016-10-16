//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#ifndef TX1DNN_OPENCV_FACE_DETECTOR_H
#define TX1DNN_OPENCV_FACE_DETECTOR_H

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "common/common.h"
#include "processor.h"

class OpenCVFaceDetector : public Processor {
 public:
  // TODO: Use a configurable path
  OpenCVFaceDetector(
      std::shared_ptr<Stream> input_stream,
      string classifier_xml_path =
          "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string classifier_xml_path_;
#ifdef USE_CUDA
  cv::gpu::CascadeClassifier_GPU classifier_;
#else
  cv::CascadeClassifier classifier_;
#endif
};

#endif  // TX1DNN_OPENCV_FACE_DETECTOR_H