/**
 * @brief rpc_receiver.cpp - Subscribe to frames over ZMQ
 */

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "processor/pubsub/frame_subscriber.h"
#include "utils/image_utils.h"

namespace po = boost::program_options;

/**
 * @brief Subscribe to frames over ZMQ and display them
 *
 * Pipeline:  FrameSubscriber -> Display
 */
void Run(std::string& server, float zoom, unsigned int angle) {
  // Set up FrameSubscriber (receives frames over ZMQ)
  auto subscriber = new FrameSubscriber(server);
  auto reader = subscriber->GetSink()->Subscribe();

  // Set up OpenCV display window
  const auto window_name = "output";
  cv::namedWindow(window_name);
  subscriber->Start();

  std::cout << "Press \"q\" to stop." << std::endl;

  while (true) {
    auto frame = reader->PopFrame(30);
    if (frame != nullptr) {
      const cv::Mat& img = frame->GetValue<cv::Mat>("original_image");
      cv::Mat m;
      cv::resize(img, m, cv::Size(), zoom, zoom);
      RotateImage(m, angle);
      cv::imshow(window_name, m);
    }

    unsigned char q = cv::waitKey(10);
    if (q == 'q') break;
  }

  subscriber->Stop();
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
  desc.add_options()("rotate,r", po::value<unsigned int>()->default_value(0),
                     "Angle to rotate image; must be 0, 90, 180, or 270");
  desc.add_options()("zoom,z", po::value<float>()->default_value(1.0),
                     "Zoom factor by which to scale image for display");
  desc.add_options()("publish_url,s",
                     po::value<string>()->default_value("127.0.0.1:5536"),
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
  auto zoom = vm["zoom"].as<float>();

  auto angles = std::set<unsigned int>{0, 90, 180, 270};
  auto angle = vm["rotate"].as<unsigned int>();
  if (!angles.count(angle)) {
    std::cerr << "--rotate angle must be 0, 90, 180, or 270" << std::endl;
    std::cerr << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  Run(server, zoom, angle);
}
