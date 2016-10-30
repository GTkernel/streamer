//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include <iostream>
#include "streamer.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  CameraManager &camera_manager = CameraManager::GetInstance();

  if (string(argv[1]) == "-h" || argc < 2) {
    std::cerr << "THIS IS WORK IN PROGRESS\n";
    std::cerr << "Usage: face CAMERA DISPLAY?\n"
              << "\n"
              << " CAMERA: The name of the camera\n"
              << " DISPLAY?: Enable preview or not\n";
  }

  string camera_name = argv[1];
  string display = argv[2];
  bool display_on = (display == "true");

  auto camera = camera_manager.GetCamera(camera_name);
  auto camera_stream = camera->GetStream();

  OpenCVFaceDetector face_detector(camera_stream);
  camera->Start();
  face_detector.Start();

  auto output_stream = face_detector.GetSinks()[0];
  auto output_reader = output_stream->Subscribe();
  if (display_on) cv::namedWindow("Image");
  while (true) {
    auto frame = output_reader->PopFrame<MetadataFrame>();
    cv::Mat image = frame->GetOriginalImage();
    auto results = frame->GetBboxes();
    cv::Scalar box_color(255, 0, 0);
    for (auto result : results) {
      cv::Rect rect(result.px, result.py, result.width, result.height);
      cv::rectangle(image, rect, box_color, 2);
    }
    if (display_on) cv::imshow("Image", image);
    int q = cv::waitKey(10);
    if (q == 'q') {
      break;
    }
  }

  output_reader->UnSubscribe();
  face_detector.Stop();
  camera->Stop();
}
