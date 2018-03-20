//
// Created by Abhinav Garlapati (abhinav2710@gmail.com) on 1/21/16.
//

#include <iostream>
#include "streamer.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init stramer context, this must be called before usign streamer.
  Context::GetContext().Init();

  CameraManager& camera_manager = CameraManager::GetInstance();

  if (std::string(argv[1]) == "-h" || argc < 2) {
    std::cerr << "Usage: ./people CAMERA DISPLAY \n"
              << "\n"
              << " CAMERA: The name of the camera\n"
              << " DISPLAY: Enable preview or not (true)\n";
  }

  std::string camera_name = argv[1];
  std::string display = argv[2];
  bool display_on = (display == "true");

  auto camera = camera_manager.GetCamera(camera_name);
  auto camera_stream = camera->GetStream();

  // This transformer is here purely as an adaptor between the camera and the
  // people detector. The 200x200 size is arbitrary.
  auto transformer = std::make_shared<ImageTransformer>(Shape(200, 200), false);
  transformer->SetSource(camera_stream);

  OpenCVPeopleDetector people_detector;

  people_detector.SetSource("input", transformer->GetSink());
  camera->Start();
  people_detector.Start();

  auto output_stream = people_detector.GetSink("output");
  auto output_reader = output_stream->Subscribe();
  if (display_on) cv::namedWindow("Image");
  while (true) {
    auto frame = output_reader->PopFrame();
    auto image = frame->GetValue<cv::Mat>("original_image");
    auto results = frame->GetValue<std::vector<Rect>>("bounding_boxes");
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
}
