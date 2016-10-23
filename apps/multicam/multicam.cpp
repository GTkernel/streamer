/**
 * @brief multicam.cpp - An example showing the usage to run realtime
 * classification on multiple camera streams.
 */

#include "tx1dnn.h"
#include "utils/utils.h"

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
  ModelManager &model_manager = ModelManager::GetInstance();

  if (argc < 4) {
    std::cout << argv[0] << " - multi-camera classification example\n"
              << "Usage: \n"
              << " CAMERAS\n"
              << " MODEL\n"
              << " DISPLAY\n";
    std::cout << std::endl;
    std::cout
        << "  CAMERAS: comma separated names of the camera in the config file\n"
        << "  MODEL: name of the model (in the config file) to use\n"
        << "  DISPLAY: display the frame or not, must have a X window if "
           "display is enabled\n";
    exit(1);
  }

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
  std::vector<std::shared_ptr<Stream>> camera_streams;
  for (auto camera : cameras) {
    auto camera_stream = camera->GetStream();
    camera_streams.push_back(camera_stream);
  }

  // Processor
  Shape input_shape(3, 227, 227);
  std::vector<std::shared_ptr<Stream>> input_streams;
  std::vector<std::shared_ptr<Processor>> processors;

  for (auto camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(new ImageTransformProcessor(
        camera_stream, input_shape, CROP_TYPE_CENTER,
        true /* subtract mean */));
    processors.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSinks()[0]);
  }

  auto model_desc = model_manager.GetModelDesc(model_name);
  std::shared_ptr<ImageClassificationProcessor> classifier(
      new ImageClassificationProcessor(input_streams, model_desc, input_shape));
  processors.push_back(classifier);

  if (display) {
    for (string camera_name : camera_names) {
      cv::namedWindow(camera_name);
    }
  }

  for (auto camera : cameras) {
    camera->Start();
  }

  for (auto processor : processors) {
    processor->Start();
  }

  int update_overlay = 0;
  const int UPDATE_OVERLAY_INTERVAL = 10;
  string label_to_show = "XXX";
  double fps_to_show = 0.0;
  while (true) {
    for (int i = 0; i < camera_names.size(); i++) {
      auto stream = classifier->GetSinks()[i];
      auto md_frame = stream->PopMDFrame();
      if (display) {
        cv::Mat img = md_frame->GetOriginalImage();
        string label = md_frame->GetTags()[0];
        if (update_overlay == 1) {
          label_to_show = label;
          fps_to_show = classifier->GetAvgFps();
        }

        // Overlay FPS label and classification label
        double font_size = 0.8 * img.size[0] / 320.0;
        cv::Point label_point(img.rows / 6, img.cols / 3);
        cv::Scalar outline_color(0, 0, 0);
        cv::Scalar label_color(200, 200, 250);

        cv::putText(img, label_to_show, label_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, outline_color, 8, CV_AA);
        cv::putText(img, label_to_show, label_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, label_color, 2, CV_AA);

        cv::Point fps_point(img.rows / 3, img.cols / 6);

        char fps_string[256];
        sprintf(fps_string, "%.2lffps", fps_to_show);
        cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, outline_color, 8, CV_AA);
        cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, label_color, 2, CV_AA);

        cv::imshow(camera_names[i], img);
      }
    }
    int q = cv::waitKey(10);
    if (q == 'q') break;
    update_overlay = (update_overlay + 1) % UPDATE_OVERLAY_INTERVAL;
  }

  LOG(INFO) << "Done";

  for (auto processor : processors) {
    processor->Stop();
  }

  for (auto camera : cameras) {
    camera->Stop();
  }

  cv::destroyAllWindows();

  return 0;
}
