//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include "opencv_face_detector.h"

OpenCVFaceDetector::OpenCVFaceDetector(string classifier_xml_path)
    : Processor(PROCESSOR_TYPE_OPENCV_FACE_DETECTOR, {"input"}, {"output"}),
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
  auto frame = GetFrame<ImageFrame>("input");
  cv::Mat image = frame->GetImage();

  std::vector<cv::Rect> results;

  std::vector<Rect> results_rect;

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
  }
#else
  classifier_.detectMultiScale(image, results);
  for (const auto& result : results) {
    results_rect.emplace_back(result.x, result.y, result.width, result.height);
  }
#endif  // USE_CUDA

  PushFrame("output",
            new MetadataFrame(results_rect, frame->GetOriginalImage()));
}
