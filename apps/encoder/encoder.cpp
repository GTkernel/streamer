/**
 * @brief encoder.cpp - An example application showing the usage of encoder.
 */

#include "tx1dnn.h"

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  CameraManager &camera_manager = CameraManager::GetInstance();

  if (argc < 3) {
    std::cout << argv[0]
              << " - Encode live streams from camera to a video file\n"
              << "Usage: \n"
              << "  " << argv[0] << " CAMERA FILE\n";
    std::cout << std::endl;
    std::cout << "  CAMERA: The name of the camera to read from\n"
              << "  FILE: The file to store the encoded result\n";
    exit(1);
  }

  // Get options
  string camera_name = argv[1];
  string dst_file = argv[2];

  CHECK(camera_manager.HasCamera(camera_name)) << "Camera " << camera_name
                                               << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);
  auto camera_stream = camera->GetStream();

  // Encoder
  GstVideoEncoder encoder(camera_stream, 640, 480, dst_file);

  camera->Start();

  encoder.Start();

  std::cout << "Press any key to stop" << std::endl;
  getchar();

  encoder.Stop();
  camera->Stop();

  return 0;
}
