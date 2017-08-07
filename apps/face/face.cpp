//
// Created by Ran Xian (xranthoar@gmail.com) on 10/11/16.
//

#include <cstdio>
#include <iostream>

#include "streamer.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  CameraManager& camera_manager = CameraManager::GetInstance();

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

  OpenCVFaceDetector face_detector;
  face_detector.SetSource("input", camera_stream);
  camera->Start();
  face_detector.Start();

  if (display_on) {
    std::cout << "Press \"q\" to stop." << std::endl;

    auto output_stream = face_detector.GetSink("output");
    auto output_reader = output_stream->Subscribe();
    cv::namedWindow("Image");
    while (true) {
      auto frame = output_reader->PopFrame();
      cv::Mat image = frame->GetValue<cv::Mat>("original_image");
      auto results = frame->GetValue<std::vector<Rect>>("bounding_boxes");
      cv::Scalar box_color(255, 0, 0);
      for (const auto& result : results) {
        cv::Rect rect(result.px, result.py, result.width, result.height);
        cv::rectangle(image, rect, box_color, 2);
      }
      cv::imshow("Image", image);
      int q = cv::waitKey(10);
      if (q == 'q') {
        break;
      }
    }

    output_reader->UnSubscribe();
  } else {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
  }

  face_detector.Stop();
  camera->Stop();
}
