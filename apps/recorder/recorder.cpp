
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "processor/image_transformer.h"
#include "processor/processor.h"
#include "processor/pubsub/frame_subscriber.h"
#include "utils/file_utils.h"
#include "video/gst_video_encoder.h"

namespace po = boost::program_options;

void Run(bool use_camera, const std::string& camera_name,
         const std::string& publish_url, unsigned int angle, bool resize,
         int x_dim, int y_dim, const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  StreamPtr stream;
  if (use_camera) {
    // Create Camera.
    std::shared_ptr<Camera> camera =
        CameraManager::GetInstance().GetCamera(camera_name);
    procs.push_back(camera);
    stream = camera->GetStream();
  } else {
    // Create FrameSubscriber.
    auto subscriber = std::make_shared<FrameSubscriber>(publish_url);
    procs.push_back(subscriber);
    stream = subscriber->GetSink();
  }

  if (resize) {
    // Create ImageTransformer.
    auto transformer =
        std::make_shared<ImageTransformer>(Shape(3, x_dim, y_dim), true, angle);
    transformer->SetSource(stream);
    procs.push_back(transformer);
    stream = transformer->GetSink();
  }

  std::string field;
  if (resize) {
    // The ImageTransformer is hardcorded to store the resized image at the key
    // "image".
    field = "image";
  } else {
    field = "original_image";
  }

  // Create GstVideoEncoder.
  std::ostringstream filepath;
  filepath << output_dir << "/" << camera_name << ".mp4";
  auto encoder = std::make_shared<GstVideoEncoder>(field, filepath.str());
  encoder->SetSource(stream);
  procs.push_back(encoder);

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
  po::options_description desc("Stores a stream as an MP4 file.");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>(),
                     "The name of the camera to use. Overrides "
                     "\"--publish-url\".");
  desc.add_options()("publish-url,u", po::value<std::string>(),
                     "The URL (host:port) on which the frame stream is being "
                     "published.");
  desc.add_options()("rotate,r", po::value<unsigned int>()->default_value(0),
                     "The angle to rotate frames; must be 0, 90, 180, or 270.");
  desc.add_options()("x-dim,x", po::value<int>(),
                     "The width to which to resize the frames.");
  desc.add_options()("y-dim,y", po::value<int>(),
                     "The height to which to resize the frames.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to store the frame JPEGs.");

  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

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

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  std::string camera;
  bool use_camera = args.count("camera");
  if (use_camera) {
    camera = args["use_camera"].as<std::string>();
  }
  std::string publish_url;
  if (args.count("publish-url")) {
    publish_url = args["publish_url"].as<std::string>();
  } else if (!use_camera) {
    throw std::runtime_error(
        "Must specify either \"--camera\" or \"--publish-url\".");
  }
  auto angles = std::set<unsigned int>{0, 90, 180, 270};
  auto angle = args["rotate"].as<unsigned int>();
  if (!angles.count(angle)) {
    std::ostringstream msg;
    msg << "Value for \"--rotate\" must be 0, 90, 180, or 270, but is: "
        << angle;
    throw std::invalid_argument(msg.str());
  }
  bool resize = false;
  int x_dim = 0;
  if (args.count("x-dim")) {
    resize = true;
    x_dim = args["x-dim"].as<int>();
    if (x_dim < 1) {
      std::ostringstream msg;
      msg << "Value for \"--x-dim\" must be greater than 0, but is: " << x_dim;
      throw std::invalid_argument(msg.str());
    }
  }
  int y_dim = 0;
  if (args.count("y-dim")) {
    y_dim = args["y-dim"].as<int>();
    if (y_dim < 1) {
      std::ostringstream msg;
      msg << "Value for \"--y-dim\" must be greater than 0, but is: " << y_dim;
      throw std::invalid_argument(msg.str());
    }
    if (!resize) {
      throw std::invalid_argument(
          "\"--x-dim\" and \"--y-dim\" must be used together.");
    }
    resize = true;
  } else if (resize) {
    throw std::invalid_argument(
        "\"--x-dim\" and \"--y-dim\" must be used together.");
  }

  auto output_dir = args["output-dir"].as<std::string>();
  Run(use_camera, camera, publish_url, angle, resize, x_dim, y_dim, output_dir);
  return 0;
}
