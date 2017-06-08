/**
 * @brief rpc_sender.cpp - Send frames over RPC
 */

#include <boost/program_options.hpp>

#include "streamer.h"

namespace po = boost::program_options;

// Global arguments
struct Configurations {
  // The name of the camera to use
  string camera_name;
  // Address of the server to send frames
  string server;
  // How many seconds to send stream
  unsigned int duration;
} CONFIG;

/**
 * @brief Send frames ver RPC
 *
 * Pipeline:  Camera -> FrameSender
 */
void Run() {
  // Set up camera
  auto camera_name = CONFIG.camera_name;
  auto& camera_manager = CameraManager::GetInstance();
  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";
  auto camera = camera_manager.GetCamera(camera_name);

  // Set up FrameSender (sends frames over RPC)
  auto frame_sender = new FrameSender(CONFIG.server);
  frame_sender->SetSource(camera->GetStream());

  frame_sender->Start();
  camera->Start();

  std::this_thread::sleep_for(std::chrono::seconds(CONFIG.duration));

  camera->Stop();
  frame_sender->Stop();
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
  desc.add_options()("server_url,s", po::value<string>()->required(),
                     "host:port to connect to (e.g., example.com:4444)");
  desc.add_options()("camera,c", po::value<string>()->required(),
                     "The name of the camera to use");
  desc.add_options()("duration,d", po::value<unsigned int>()->default_value(5),
                     "How long to send stream in seconds");

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

  CONFIG.server = vm["server_url"].as<string>();
  CONFIG.camera_name = vm["camera"].as<string>();
  CONFIG.duration = vm["duration"].as<unsigned int>();

  Run();
}
