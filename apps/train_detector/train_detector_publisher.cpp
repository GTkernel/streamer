/**
 * @brief train_detector_publisher.cpp - This application throttles a camera
 * stream and publishes it on the network.
 */

#include <csignal>

#include <boost/program_options.hpp>

#include "processor/pubsub/frame_publisher.h"
#include "processor/throttler.h"
#include "streamer.h"

namespace po = boost::program_options;

std::shared_ptr<Camera> camera;
std::shared_ptr<Throttler> throttler;
std::shared_ptr<FramePublisher> publisher;

void CleanUp() {
  // Stop Processors in forward order
  if (camera != nullptr && camera->IsStarted()) camera->Stop();
  if (throttler != nullptr && throttler->IsStarted()) throttler->Stop();
  if (publisher != nullptr && publisher->IsStarted()) publisher->Stop();
}

void SignalHandler(int) {
  LOG(INFO) << "Received SIGINT, trying to exit gracefully...";
  CleanUp();
  exit(0);
}

void Run(const std::string& camera_name, int fps,
         const std::string& publish_url) {
  // Set up camera
  auto& camera_manager = CameraManager::GetInstance();
  camera = camera_manager.GetCamera(camera_name);

  // Set up Throttler (decimates stream to target FPS)
  throttler = std::make_shared<Throttler>(fps);
  throttler->SetSource("input", camera->GetStream());

  // Set up FramePublisher (publishes frames via ZMQ)
  publisher = std::make_shared<FramePublisher>(publish_url);
  publisher->SetSource(throttler->GetSink("output"));

  // Start Processors in reverse order
  publisher->Start();
  throttler->Start();
  camera->Start();

  while (true) {
  }
}

int main(int argc, char* argv[]) {
  std::signal(SIGINT, SignalHandler);

  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple Frame Sender App for Streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("fps,f", po::value<int>()->required(),
                     ("The maximum rate of the published stream. The actual "
                      "rate may be less."));
  desc.add_options()("publish_url,u",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "host:port to connect to (e.g., example.com:4444)");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  //// Parse arguments
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<std::string>());
  }
  auto camera_name = vm["camera"].as<std::string>();
  auto fps = vm["fps"].as<int>();
  auto publish_url = vm["publish_url"].as<std::string>();

  // Init streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  Run(camera_name, fps, publish_url);
  return 0;
}
