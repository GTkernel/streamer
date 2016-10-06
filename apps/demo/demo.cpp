//
// Created by Ran Xian (xranthoar@gmail.com) on 10/5/16.
//

#include "tx1dnn.h"

CameraManager &camera_manager = CameraManager::GetInstance();
ModelManager &model_manager = ModelManager::GetInstance();

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  if (argc != 5) {
    std::cout << "Usage:\n"
              << " CAMERA\n"
              << " classify|segment\n"
              << " MODEL\n"
              << " DISPLAY\n";
    std::cout << std::endl;
    std::cout << " CAMERA: the name of the camera in the config file\n"
              << " classify: run image classificaiton, segment: run image "
                 "segmentation\n"
              << " MODEL: the name of the model in the config file\n"
              << " DISPLAY: display the frame or not, must have a X window if "
                 "display is enabled\n";
    exit(1);
  }

  // Set up
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  // Get options
  string camera_name = argv[1];
  string task = argv[2];
  string model_name = argv[3];
  string display_on = argv[4];

  // Check options
  CHECK(model_manager.HasModel(model_name)) << "Model " << model_name
                                            << " does not exist";
  CHECK(camera_manager.HasCamera(camera_name)) << "Camera " << camera_name
                                               << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);
  bool display = (display_on == "true");

  // Do video stream classification
  LOG(INFO) << "Do video stream " << task << " on " + camera_name;
  if (display) {
    cv::namedWindow("camera");
  }

  auto camera_stream = camera->GetStream();

  // Processor
  cv::namedWindow("Camera");
  cv::namedWindow("Result");
  camera->Start();
  if (task == "classification") {
    Shape input_shape(3, 227, 227);
    ImageTransformProcessor transform_processor = ImageTransformProcessor(
        camera_stream, input_shape, CROP_TYPE_CENTER, true /* subtract mean */);

    auto model_desc = model_manager.GetModelDesc(model_name);
    ImageClassificationProcessor classification_processor(
        transform_processor.GetSinks()[0], transform_processor.GetSinks()[1],
        model_desc, input_shape);

    transform_processor.Start();
    classification_processor.Start();

    auto output_stream = classification_processor.GetSinks()[0];
    string user_input;
    while (true) {
      cv::Mat frame = output_stream->PopFrame();
      cv::imshow("Camera", frame);
      char k = cv::waitKey(10);
      if (k == 'q') {
        break;
      }
    }

    classification_processor.Stop();
    transform_processor.Stop();
  } else {
    Shape input_shape(3, 250, 250);
    ImageTransformProcessor transform_processor = ImageTransformProcessor(
        camera_stream, input_shape, CROP_TYPE_CENTER, true /* subtract mean */);

    auto model_desc = model_manager.GetModelDesc(model_name);
    ImageSegmentationProcessor segmentation_processor(
        transform_processor.GetSinks()[0], transform_processor.GetSinks()[1],
        model_desc, input_shape);

    transform_processor.Start();
    segmentation_processor.Start();

    auto seg_stream = segmentation_processor.GetSinks()[0];
    auto img_stream = segmentation_processor.GetSinks()[1];

    string user_input;
    while (true) {
      cv::Mat result = seg_stream->PopFrame();
      LOG(INFO) << "result " << result.size[0] << " " << result.size[1];
      cv::Mat frame = img_stream->PopFrame();
      LOG(INFO) << frame.size;
      cv::imshow("Result", result);
      cv::imshow("Camera", frame);
      char k = cv::waitKey(10);
      if (k == 'q') {
        break;
      }
    }

    segmentation_processor.Stop();
    transform_processor.Stop();
  }

  camera->Stop();

  cv::destroyAllWindows();

  return 0;
}