//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include "opencv_face_detector.h"

OpenCVFaceDetector::OpenCVFaceDetector(float idle_duration,
                                       string classifier_xml_path)
    : Processor(PROCESSOR_TYPE_OPENCV_FACE_DETECTOR, {"input"}, {"output"}),
      idle_duration_(idle_duration),
      classifier_xml_path_(classifier_xml_path) {}

std::shared_ptr<OpenCVFaceDetector> OpenCVFaceDetector::Create(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool OpenCVFaceDetector::Init() {
  return classifier_.load(classifier_xml_path_);
}

bool OpenCVFaceDetector::OnStop() {
  classifier_.empty();
  return true;
}

void OpenCVFaceDetector::Process() {
  auto frame = GetFrame("input");
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - last_detect_time_;
  if (diff.count() >= idle_duration_) {
    const cv::Mat& image = frame->GetValue<cv::Mat>("image");

    std::vector<cv::Rect> results;

    std::vector<Rect> results_rect;
    std::vector<std::string> tags;

#ifdef USE_CUDA
    cv::gpu::GpuMat image_gpu(image);
    cv::gpu::GpuMat faces;
    int num_face = classifier_.detectMultiScale(image_gpu, faces);
    cv::Mat obj_host;
    faces.colRange(0, num_face).download(obj_host);
    cv::Rect* cfaces = obj_host.ptr<cv::Rect>();
    for (decltype(num_face) i = 0; i < num_face; ++i) {
      results_rect.emplace_back(cfaces[i].x, cfaces[i].y, cfaces[i].width,
                                cfaces[i].height);
      tags.push_back("face");
    }
#else
    classifier_.detectMultiScale(image, results);
    for (const auto& result : results) {
      results_rect.emplace_back(result.x, result.y, result.width,
                                result.height);
      tags.push_back("face");
    }
#endif  // USE_CUDA

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("bounding_boxes", results_rect);
    frame->SetValue("tags", tags);
    PushFrame("output", std::move(frame));
  } else {
    PushFrame("output", std::move(frame));
  }
}
