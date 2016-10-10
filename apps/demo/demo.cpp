//
// Created by Ran Xian (xranthoar@gmail.com) on 10/5/16.
//

#include "tx1dnn.h"

CameraManager &camera_manager = CameraManager::GetInstance();
ModelManager &model_manager = ModelManager::GetInstance();

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  if (argc != 5) {
    std::cout << "Usage:\n"
              << " CAMERA\n"
              << " MODEL\n"
              << " DISPLAY\n";
    std::cout << std::endl;
    std::cout << " CAMERA: the name of the camera in the config file\n"
              << " MODEL: the name of the model in the config file\n"
              << " DISPLAY: display the frame or not, must have a X window if "
                 "display is enabled\n";
    exit(1);
  }

  // Get options
  string camera_name = argv[1];
  string model_name = argv[2];
  string display_on = argv[3];

  // Check options
  CHECK(model_manager.HasModel(model_name)) << "Model " << model_name
                                            << " does not exist";
  CHECK(camera_manager.HasCamera(camera_name)) << "Camera " << camera_name
                                               << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);
  bool display = (display_on == "true");

  // Do video stream classification
  LOG(INFO) << "Do video stream segmentation on " + camera_name;
  if (display) {
    cv::namedWindow("camera");
  }

  auto camera_stream = camera->GetStream();

  // Processor
  cv::namedWindow("Camera");
  camera->Start();

  if (display) {
    cv::namedWindow("Result");
  }
  Shape input_shape(3, 250, 250);
  ImageTransformProcessor transform_processor = ImageTransformProcessor(
      camera_stream, input_shape, CROP_TYPE_CENTER, true /* subtract mean */);

  auto model_desc = model_manager.GetModelDesc(model_name);
  ImageSegmentationProcessor segmentation_processor(
      transform_processor.GetSinks()[0], model_desc, input_shape);

  transform_processor.Start();
  segmentation_processor.Start();

  auto seg_stream = segmentation_processor.GetSinks()[0];

  while (true) {
    auto frame = seg_stream->PopImageFrame();
    if (display) {
      cv::imshow("Result", frame->GetImage());
      cv::imshow("Camera", frame->GetOriginalImage());
      int k = cv::waitKey(10);
      if (k == 'q') {
        break;
      }
    }
  }

  segmentation_processor.Stop();
  transform_processor.Stop();

  camera->Stop();

  cv::destroyAllWindows();

  return 0;
}