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
#include "object_detector.h"

class OpenCVFaceDetector : public BaseDetector {
 public:
  OpenCVFaceDetector(const ModelDesc& model_desc)
    : model_desc_(model_desc) {}
  virtual ~OpenCVFaceDetector() {}
  virtual bool Init();
  virtual std::vector<ObjectInfo> Detect(const cv::Mat& image);

 private:
  ModelDesc model_desc_;
#ifdef USE_CUDA
  cv::cuda::CascadeClassifier classifier_;
#else
  cv::CascadeClassifier classifier_;
#endif  // USE_CUDA
};

#endif  // STREAMER_PROCESSOR_OPENCV_FACE_DETECTOR_H_
