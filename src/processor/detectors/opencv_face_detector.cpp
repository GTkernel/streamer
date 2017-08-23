//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include "processor/detectors/opencv_face_detector.h"

bool OpenCVFaceDetector::Init() {
  return classifier_.load(model_desc_.GetModelParamsPath());
}

std::vector<ObjectInfo> OpenCVFaceDetector::Detect(const cv::Mat& image) {
  std::vector<ObjectInfo> result;

#ifdef USE_CUDA
  cv::gpu::GpuMat image_gpu(image);
  cv::gpu::GpuMat faces;
  int num_face = classifier_.detectMultiScale(image_gpu, faces);
  cv::Mat obj_host;
  faces.colRange(0, num_face).download(obj_host);
  cv::Rect* cfaces = obj_host.ptr<cv::Rect>();
  for (decltype(num_face) i = 0; i < num_face; ++i) {
    ObjectInfo object_info;
    object_info.tag = "face";
    object_info.bbox = cfaces[i];
    object_info.confidence = 1.0;
    result.push_back(object_info);
  }
#else
  std::vector<cv::Rect> rects;
  classifier_.detectMultiScale(image, rects);
  for (const auto& m : rects) {
    ObjectInfo object_info;
    object_info.tag = "face";
    object_info.bbox = m;
    object_info.confidence = 1.0;
    result.push_back(object_info);
  }
#endif  // USE_CUDA

  return result;
}
