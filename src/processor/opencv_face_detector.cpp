//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include "opencv_face_detector.h"

OpenCVFaceDetector::OpenCVFaceDetector(std::shared_ptr<Stream> input_stream,
                                       string classifier_xml_path)
    : classifier_xml_path_(classifier_xml_path) {
  sources_.push_back(input_stream);
  sinks_.emplace_back(new Stream("OpenCVFaceDetector"));
}

bool OpenCVFaceDetector::Init() {
  return classifier_.load(classifier_xml_path_);
}

bool OpenCVFaceDetector::OnStop() {
  classifier_.empty();
  return true;
}

void OpenCVFaceDetector::Process() {
  auto frame = sources_[0]->PopImageFrame();
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
  for (int i = 0; i < num_face; i++) {
    results_rect.emplace_back(cfaces[i].x, cfaces[i].y, cfaces[i].width,
                              cfaces[i].height);
  }
#else
  classifier_.detectMultiScale(image, results);
  for (auto result : results) {
    results_rect.emplace_back(result.x, result.y, result.width, result.height);
  }
#endif

  sinks_[0]->PushFrame(
      new MetadataFrame(results_rect, frame->GetOriginalImage()));
}
