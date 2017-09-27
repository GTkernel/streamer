// This application throttles a camera stream and publishes it on the network.

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "processor/pubsub/frame_publisher.h"
#include "processor/throttler.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, int fps,
         const std::string& publish_url) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  auto& camera_manager = CameraManager::GetInstance();
  auto camera = camera_manager.GetCamera(camera_name);
  procs.push_back(camera);

  StreamPtr frame_stream = camera->GetStream();
  if (fps) {
    // Create Throttler (decimates stream to target FPS).
    auto throttler = std::make_shared<Throttler>(fps);
    throttler->SetSource("input", frame_stream);
    procs.push_back(throttler);
    frame_stream = throttler->GetSink("output");
  }

  // Create FramePublisher (publishes frames via ZMQ).
  auto publisher = std::make_shared<FramePublisher>(publish_url);
  publisher->SetSource(frame_stream);
  procs.push_back(publisher);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Simple Frame Sender App for Streamer");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("fps,f", po::value<int>()->default_value(0),
                     ("The maximum rate of the published stream. The actual "
                      "rate may be less. An fps of 0 disables throttling."));
  desc.add_options()(
      "publish_url,u",
      po::value<std::string>()->default_value("127.0.0.1:5536"),
      "host:port to publish the stream on (e.g., 127.0.0.1:5536)");

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

  // Set up GStreamer.
  gst_init(&argc, &argv);
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  std::string camera_name = args["camera"].as<std::string>();
  int fps = args["fps"].as<int>();
  std::string publish_url = args["publish_url"].as<std::string>();
  Run(camera_name, fps, publish_url);
  return 0;
}
