//
// Created by Ran Xian (xranthoar@gmail.com) on 10/5/16.
//

#include <cstdio>
#include "processor/image_transformer.h"
#include "processor/image_segmenter.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "camera/camera_manager.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  if (argc != 4) {
    std::cout << argv[0] << " - Image segmentation example\n";
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
  std::string camera_name = argv[1];
  std::string model_name = argv[2];
  std::string display_on = argv[3];

  // Check options
  CHECK(model_manager.HasModel(model_name))
      << "Model " << model_name << " does not exist";
  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);

  // Do video stream classification
  LOG(INFO) << "Do video stream segmentation on " + camera_name;

  // Processor
  camera->Start();

  auto camera_stream = camera->GetStream();

  Shape input_shape(3, 250, 250);
  ImageTransformer transform_processor(input_shape, true, true);
  transform_processor.SetSource("input", camera_stream);

  auto model_desc = model_manager.GetModelDesc(model_name);
  ImageSegmenter segmentation_processor(model_desc, input_shape);
  segmentation_processor.SetSource("input",
                                   transform_processor.GetSink("output"));

  transform_processor.Start();
  segmentation_processor.Start();

  bool display = (display_on == "true");
  if (display) {
    std::cout << "Press \"q\" to stop." << std::endl;

    auto seg_stream = segmentation_processor.GetSink("output");
    auto reader = seg_stream->Subscribe();

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::namedWindow("Result", cv::WINDOW_NORMAL);

    while (true) {
      auto frame = reader->PopFrame();
      cv::imshow("Result", frame->GetValue<cv::Mat>("image"));
      cv::imshow("Camera", frame->GetValue<cv::Mat>("original_image"));
      int k = cv::waitKey(10);
      if (k == 'q') {
        break;
      }
    }

    reader->UnSubscribe();
  } else {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
  }

  segmentation_processor.Stop();
  transform_processor.Stop();

  camera->Stop();

  cv::destroyAllWindows();

  return 0;
}
