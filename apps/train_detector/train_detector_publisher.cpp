// This application throttles a camera stream and publishes it on the network.

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "processor/pubsub/frame_publisher.h"
#include "processor/throttler.h"

namespace po = boost::program_options;

std::vector<std::shared_ptr<Processor>> procs;

void SignalHandler(int) {
  LOG(INFO) << "Received SIGINT, stopping...";

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    if (proc->IsStarted()) proc->Stop();
  }
  exit(0);
}

void Run(const std::string& camera_name, int fps,
         const std::string& publish_url) {
  // Create Camera.
  auto& camera_manager = CameraManager::GetInstance();
  auto camera = camera_manager.GetCamera(camera_name);
  procs.push_back(camera);

  // Create Throttler (decimates stream to target FPS).
  auto throttler = std::make_shared<Throttler>(fps);
  throttler->SetSource("input", camera->GetStream());
  procs.push_back(throttler);

  // Create FramePublisher (publishes frames via ZMQ).
  auto publisher = std::make_shared<FramePublisher>(publish_url);
  publisher->SetSource(throttler->GetSink("output"));
  procs.push_back(publisher);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  while (true) continue;
}

int main(int argc, char* argv[]) {
  po::options_description desc("Simple Frame Sender App for Streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("fps,f", po::value<int>()->required(),
                     ("The maximum rate of the published stream. The actual "
                      "rate may be less."));
  desc.add_options()(
      "publish_url,u",
      po::value<std::string>()->default_value("127.0.0.1:5536"),
      "host:port to publish the stream on (e.g., 127.0.0.1:4444)");

  // Parse the command line arguments.
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, desc), args);
    if (args.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }
    po::notify(args);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  // Register the signal handler that will stop the processors.
  std::signal(SIGINT, SignalHandler);
  // Set up GStreamer.
  gst_init(&argc, &argv);
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  std::string camera_name = args["camera"].as<std::string>();
  int fps = args["fps"].as<int>();
  std::string publish_url = args["publish_url"].as<std::string>();
  Run(camera_name, fps, publish_url);
  return 0;
}
