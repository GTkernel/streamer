//
// Created by Abhinav Garlapati (abhinav2710@gmail.com) on 1/21/17.
//

#include "opencv_people_detector.h"

OpenCVPeopleDetector::OpenCVPeopleDetector()
    : Processor(PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR, {"input"}, {"output"}) {}

bool OpenCVPeopleDetector::Init() {
  hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  return true;
}

bool OpenCVPeopleDetector::OnStop() {
  // hog_.empty();
  return true;
}

void OpenCVPeopleDetector::Process() {
  auto frame = GetFrame("input");
  const cv::Mat& image = frame->GetValue<cv::Mat>("image");

  std::vector<cv::Rect> results;

  std::vector<Rect> results_rect;

#ifdef USE_CUDA
  cv::gpu::GpuMat image_gpu(image);

  hog_.detectMultiScale(image_gpu, results, 0, cv::Size(8, 8), cv::Size(32, 32),
                        1.05, 2);
#else
  hog_.detectMultiScale(image, results, 0, cv::Size(8, 8), cv::Size(32, 32),
                        1.05, 2);
#endif
  for (auto result : results) {
    results_rect.emplace_back(result.x, result.y, result.width, result.height);
  }

  frame->SetValue("bounding_boxes", results_rect);
  PushFrame("output", std::move(frame));
}
