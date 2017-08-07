/**
 * @brief classifier.cpp - An example application to classify images.
 */

#include <cstdio>

#include <boost/program_options.hpp>

#include "streamer.h"

namespace po = boost::program_options;

void Run(const string& camera_name, const string& net_name, bool display) {
  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);

  // Transformer
  Shape input_shape(3, 227, 227);
  auto transformer = std::make_shared<ImageTransformer>(
      input_shape, true, true, true /* subtract mean */);
  transformer->SetSource("input", camera->GetSink("output"));

  // Image classifier
  auto model_desc = model_manager.GetModelDesc(net_name);
  auto classifier =
      std::make_shared<ImageClassifier>(model_desc, input_shape, 1);
  classifier->SetSource("input", transformer->GetSink("output"));

  // Run
  camera->Start();
  transformer->Start();
  classifier->Start();

  if (display) {
    std::cout << "Press \"q\" to stop." << std::endl;

    auto reader = classifier->GetSink("output")->Subscribe();
    while (true) {
      auto frame = reader->PopFrame();
      auto tags = frame->GetValue<std::vector<std::string>>("tags");
      auto probs = frame->GetValue<std::vector<double>>("probabilities");
      std::string tag = tags.front();
      double prob = probs.front();

      cv::Mat img = frame->GetValue<cv::Mat>("original_image");

      // Overlay classification label and probability
      double font_size = 0.8 * img.size[0] / 320.0;
      cv::Point label_point(img.rows / 6, img.cols / 3);
      cv::Scalar outline_color(0, 0, 0);
      cv::Scalar label_color(200, 200, 250);

      cv::putText(img, tag, label_point, CV_FONT_HERSHEY_DUPLEX, font_size,
                  outline_color, 8, CV_AA);
      cv::putText(img, tag, label_point, CV_FONT_HERSHEY_DUPLEX, font_size,
                  label_color, 2, CV_AA);

      cv::Point fps_point(img.rows / 3, img.cols / 6);

      char fps_string[256];
      sprintf(fps_string, "Prob: %.2lf", prob);
      cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX, font_size,
                  outline_color, 8, CV_AA);
      cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX, font_size,
                  label_color, 2, CV_AA);

      cv::imshow(camera_name, img);

      int q = cv::waitKey(10);
      if (q == 'q') break;
    }
  } else {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
  }

  classifier->Stop();
  transformer->Stop();
  camera->Stop();
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple camera display test");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera",
                     po::value<string>()->value_name("CAMERA")->required(),
                     "The name of the camera to use");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR")->required(),
                     "The directory to find streamer's configurations");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()("net,n",
                     po::value<string>()->value_name("NET")->required(),
                     "The name of the neural net to run");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<string>();
  auto net_name = vm["net"].as<string>();
  auto display = vm.count("display") != 0;

  // Run classifier
  Run(camera_name, net_name, display);

  return 0;
}
