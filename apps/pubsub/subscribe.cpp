// subscribe.cpp - Subscribes to frames over ZMQ and displays them.

#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "processor/pubsub/frame_subscriber.h"
#include "stream/stream.h"
#include "utils/image_utils.h"

namespace po = boost::program_options;

constexpr auto FIELD_TO_DISPLAY = "original_image";

void Run(const std::string& publish_url, float zoom, unsigned int angle) {
  // Create FrameSubscriber.
  auto subscriber = std::make_shared<FrameSubscriber>(publish_url);
  StreamReader* reader = subscriber->GetSink()->Subscribe();

  // Set up OpenCV display window.
  cv::namedWindow(FIELD_TO_DISPLAY);
  subscriber->Start();

  std::cout << "Press \"q\" to stop." << std::endl;

  while (true) {
    auto frame = reader->PopFrame();
    if (frame != nullptr) {
      // Extract image and display it.
      const cv::Mat& img = frame->GetValue<cv::Mat>(FIELD_TO_DISPLAY);
      cv::Mat img_resized;
      cv::resize(img, img_resized, cv::Size(), zoom, zoom);
      RotateImage(img_resized, angle);
      cv::imshow(FIELD_TO_DISPLAY, img_resized);

      if (cv::waitKey(10) == 'q') break;
    }
  }

  reader->UnSubscribe();
  subscriber->Stop();
  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
  po::options_description desc("Simple frame subscriber  example app");
  desc.add_options()("help,h", "Print the help message");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("publish-url,p",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "The URL (host:port) on which the frame stream is being "
                     "published.");
  desc.add_options()("rotate,r", po::value<unsigned int>()->default_value(0),
                     "Angle to rotate image; Must be 0, 90, 180, or 270.");
  desc.add_options()("zoom,z", po::value<float>()->default_value(1.0),
                     "Zoom factor by which to scale image for display.");

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

  std::string publish_url = args["publish-url"].as<std::string>();
  float zoom = args["zoom"].as<float>();
  std::set<unsigned int> angles = std::set<unsigned int>{0, 90, 180, 270};
  unsigned int angle = args["rotate"].as<unsigned int>();
  if (angles.find(angle) == angles.end()) {
    std::cerr << "Error: \"--rotate\" angle must be 0, 90, 180, or 270."
              << std::endl;
    std::cerr << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }
  Run(publish_url, zoom, angle);
}
