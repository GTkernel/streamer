/**
 * @brief tx1_classify.cpp - An example showing the usage to run realtime
 * classification.
 */

#include "tx1dnn.h"
#include "utils/utils.h"

CameraManager &camera_manager = CameraManager::GetInstance();
ModelManager &model_manager = ModelManager::GetInstance();

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  if (argc != 4) {
    std::cout << "Usage: "
              << " CAMERAS\n"
              << " MODEL\n"
              << " DISPLAY\n";
    std::cout << std::endl;
    std::cout
        << "  CAMERA: the name of the camera in the config file\n"
        << "  MODELS: comma separated names of the model in the config file\n"
        << "  DISPLAY: display the frame or not, must have a X window if "
           "display is enabled\n";
    exit(1);
  }

  // Set up
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  // Get options
  auto camera_names = SplitString(argv[1], ",");
  string model_name = argv[2];
  string display_on = argv[3];

  // Check options
  CHECK(model_manager.HasModel(model_name)) << "Model " << model_name
                                            << " does not exist";
  for (auto camera_name : camera_names) {
    CHECK(camera_manager.HasCamera(camera_name)) << "Camera " << camera_name
                                                 << " does not exist";
  }

  std::vector<std::shared_ptr<Camera>> cameras;
  for (auto camera_name : camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  bool display = (display_on == "true");

  // Do video stream classification
  if (display) {
    cv::namedWindow("camera");
  }

  std::vector<std::shared_ptr<Stream>> camera_streams;
  for (auto camera : cameras) {
    auto camera_stream = camera->GetStream();
    camera_streams.push_back(camera_stream);
  }

  // Processor
  Shape input_shape(3, 227, 227);
  std::vector<std::shared_ptr<Stream>> input_streams;
  std::vector<std::shared_ptr<Stream>> img_streams;
  std::vector<std::shared_ptr<Processor>> processors;

  for (auto camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(new ImageTransformProcessor(
        camera_stream, input_shape, CROP_TYPE_CENTER,
        true /* subtract mean */));
    processors.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSinks()[0]);
    img_streams.push_back(transform_processor->GetSinks()[1]);
  }

  auto model_desc = model_manager.GetModelDesc(model_name);
  std::shared_ptr<ImageClassificationProcessor> batch_classifier(
      new ImageClassificationProcessor(input_streams, img_streams, model_desc, input_shape));
  processors.push_back(batch_classifier);

  for (string camera_name : camera_names) {
    cv::namedWindow(camera_name);
  }

  for (auto camera : cameras) {
    camera->Start();
  }

  for (auto processor : processors) {
    processor->Start();
  }

  while (true) {
    for (int i = 0; i < camera_names.size(); i++) {
      auto stream = batch_classifier->GetSinks()[i];
      cv::Mat frame = stream->PopFrame().GetImage();
      cv::imshow(camera_names[i], frame);
    }
    int q = cv::waitKey(10);
    if (q == 'q') break;
  }

  // Not stopping the processors
  cv::destroyAllWindows();

  return 0;
}
