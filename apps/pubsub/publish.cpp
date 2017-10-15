/**
 * @brief publish.cpp - Send frames over ZMQ
 */

#include <cstdio>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "processor/pubsub/frame_publisher.h"

#include "processor/throttler.h"

namespace po = boost::program_options;

/**
 * @brief Publish frames via ZMQ
 *
 * Pipeline:  Camera -> FramePublisher
 */
void Run(std::string camera_name, std::string server, double fps) {
  // Set up camera
  auto& camera_manager = CameraManager::GetInstance();
  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";
  auto camera = camera_manager.GetCamera(camera_name);

  auto throttler = new Throttler(fps);
  throttler->SetSource(camera->GetStream());

  // Set up FramePublisher (publishes frames via ZMQ)
  auto publisher = new FramePublisher(server);
  publisher->SetSource(throttler->GetSink());

  publisher->Start();
  throttler->Start();
  camera->Start();

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  camera->Stop();
  throttler->Stop();
  publisher->Stop();
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple Frame Sender App for Streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<string>(),
                     "The directory to find streamer's configuration");
  desc.add_options()("publish_url,s",
                     po::value<string>()->default_value("127.0.0.1:5536"),
                     "host:port to connect to (e.g., example.com:4444)");
  desc.add_options()("camera,c", po::value<string>()->required(),
                     "The name of the camera to use");
  desc.add_options()("fps,f", po::value<double>()->required(),
                     "The maximum framerate at which to publish.");

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

  //// Parse arguments

  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto server = vm["publish_url"].as<string>();
  auto camera_name = vm["camera"].as<string>();
  auto fps = vm["fps"].as<double>();

  Run(camera_name, server, fps);
}
