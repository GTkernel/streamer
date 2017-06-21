/**
 * @brief rpc_receiver.cpp - Receive frames over RPC
 */

#include <boost/program_options.hpp>

#include "streamer.h"

namespace po = boost::program_options;

// Global arguments
struct Configurations {
  // The server to receive frames
  string server;
} CONFIG;

/**
 * @brief Receive frames over RPC and display them
 *
 * Pipeline:  FrameReceiver -> Display
 */
void Run() {
  // Set up FrameReceiver (receives frames over RPC)
  auto frame_receiver = new FrameReceiver(CONFIG.server);
  auto reader = frame_receiver->GetSink()->Subscribe();

  // Set up OpenCV display window
  const auto window_name = "output";
  cv::namedWindow(window_name);
  frame_receiver->Start();

  while (true) {
    auto frame = reader->PopFrame(30);
    if (frame != nullptr) {
      cv::Mat img = frame->GetValue("Image",);
      cv::imshow(window_name, img);
    }

    unsigned char q = cv::waitKey(10);
    if (q == 'q') break;
  }

  frame_receiver->Stop();
  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple Frame Receiver App for Streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<string>(),
                     "The directory to find streamer's configuration");
  desc.add_options()("listen_url,l", po::value<string>()->required(),
                     "address:port to listen on (e.g., 0.0.0.0:4444)");

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

  CONFIG.server = vm["listen_url"].as<string>();

  Run();
}
