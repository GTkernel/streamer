/**
 * @brief tx1_classify.cpp - An example showing the usage to run realtime
 * classification.
 */

#include "tx1dnn.h"

CameraManager &camera_manager = CameraManager::GetInstance();
ModelManager &model_manager = ModelManager::GetInstance();

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  if (argc != 4) {
    std::cout << "Usage: "
              << " CAMERA\n"
              << " MODEL\n"
              << " DISPLAY\n";
    std::cout << std::endl;
    std::cout << "  CAMERA: the name of the camera in the config file\n"
              << "  MODEL: the name of the model in the config file\n"
              << "  DISPLAY: display the frame or not, must have a X window if display is enabled\n";
    exit(1);
  }

  // Set up
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // Get options
  string camera_name = argv[1];
  string model_name = argv[2];
  string display_on = argv[3];

  // Check options
  CHECK(model_manager.HasModel(model_name))
  << "Model " << model_name << " does not exist";
  CHECK(camera_manager.HasCamera(camera_name))
  << "Camera " << camera_name << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);
  auto model_desc = model_manager.GetModelDesc(model_name);
  Shape input_shape(3, 227, 227);

  std::unique_ptr<Classifier>
      classifier(new Classifier(model_desc, input_shape));

  bool display = (display_on == "true");

  // Do video stream classification
  LOG(INFO) << "Do video stream classification on " + camera_name;
  if (display) {
    cv::namedWindow("camera");
  }

  CHECK(camera->Open()) << "Can't open camera, check camera and pipeline uri";
  DataBuffer data_buffer(input_shape.GetSize() * sizeof(float));

  while (1) {
    // FIXME: Use something like a conditional variable to avoid busy waiting.
    cv::Mat frame = camera->Capture();
    classifier->Preprocess(frame, data_buffer);
    std::vector<Prediction>
        predictions = classifier->Classify(data_buffer, 1);
    Prediction p = predictions[0];
    LOG(INFO) << p.second << " - \""
              << p.first << "\"" << std::endl;
    if (display) {
      cv::imshow("camera", frame);
      cv::waitKey(10);
    }
  }

  camera->Close();

  return 0;
}
