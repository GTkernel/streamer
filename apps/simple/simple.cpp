/**
 * @brief simple.cpp - An example application to display camera data.
 */

#include <csignal>

#include <boost/program_options.hpp>

#include "streamer.h"
#include "utils/image_utils.h"

namespace po = boost::program_options;

std::shared_ptr<Camera> camera;

void SignalHandler(int) {
  std::cout << "Received SIGINT, stopping" << std::endl;
  if (camera != nullptr) camera->Stop();

  exit(0);
}

void Run(const string& camera_name, float zoom, unsigned int angle) {
  CameraManager& camera_manager = CameraManager::GetInstance();

  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  camera = camera_manager.GetCamera(camera_name);
  auto reader = camera->GetStream()->Subscribe();
  cv::namedWindow(camera_name);

  camera->Start();

  while (true) {
    auto frame = reader->PopFrame();
    if (frame != nullptr) {
      const cv::Mat& img = frame->GetValue<cv::Mat>("original_image");
      cv::Mat m;
      cv::resize(img, m, cv::Size(), zoom, zoom);
      RotateImage(m, angle);
      cv::imshow(camera_name, m);

      std::cout << frame->ToString() << std::endl;
    }

    unsigned char q = cv::waitKey(10);
    if (q == 'q') break;
  }

  camera->Stop();
}

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
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
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");
  desc.add_options()("rotate,r", po::value<unsigned int>()->default_value(0),
                     "Angle to rotate image; must be 0, 90, 180, or 270");
  desc.add_options()("zoom,z", po::value<float>()->default_value(1.0),
                     "Zoom factor by which to scale image for display");

  std::signal(SIGINT, SignalHandler);

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

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<string>();
  auto zoom = vm["zoom"].as<float>();

  auto angles = std::set<unsigned int>{0, 90, 180, 270};
  auto angle = vm["rotate"].as<unsigned int>();
  if (!angles.count(angle)) {
    std::cerr << "--rotate angle must be 0, 90, 180, or 270" << std::endl;
    std::cerr << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  Run(camera_name, zoom, angle);

  return 0;
}
