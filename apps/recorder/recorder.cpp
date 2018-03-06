
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
#include "utils/file_utils.h"
#include "video/gst_video_encoder.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, unsigned int angle, bool resize,
         int x_dim, int y_dim, const std::string& field,
         const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  StreamPtr stream = camera->GetStream();
  if (resize) {
    // Create ImageTransformer.
    auto transformer =
        std::make_shared<ImageTransformer>(Shape(3, x_dim, y_dim), true, angle);
    transformer->SetSource(camera->GetStream());
    procs.push_back(transformer);
    stream = transformer->GetSink();
  }

  std::string field_to_save;
  if (resize) {
    // The ImageTransformer is hardcorded to store the resized image at the key
    // "image".
    field_to_save = "image";
  } else {
    field_to_save = field;
  }

  // Create GstVideoEncoder.
  std::ostringstream filepath;
  filepath << output_dir << "/" << camera_name << "_" << field << ".mp4";
  auto encoder =
      std::make_shared<GstVideoEncoder>(field_to_save, filepath.str());
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
  po::options_description desc("Stores frames as JPEG images");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("rotate,r", po::value<unsigned int>()->default_value(0),
                     "The angle to rotate frames; must be 0, 90, 180, or 270.");
  desc.add_options()("x-dim,x", po::value<int>(),
                     "The width to which to resize the frames.");
  desc.add_options()("y-dim,y", po::value<int>(),
                     "The height to which to resize the frames.");
  desc.add_options()("field,f",
                     po::value<std::string>()->default_value("original_image"),
                     "The field to save as a JPEG. Assumed to be "
                     "\"original_image\" when using \"--resize\".");
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

  auto camera = args["camera"].as<std::string>();
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
      msg << "Value for \"--x-dim\ must be greater than 0, but is: " << x_dim;
      throw std::invalid_argument(msg.str());
    }
  }
  int y_dim = 0;
  if (args.count("y-dim")) {
    resize = true;
    y_dim = args["y-dim"].as<int>();
    if (y_dim < 1) {
      std::ostringstream msg;
      msg << "Value for \"--y-dim\ must be greater than 0, but is: " << y_dim;
      throw std::invalid_argument(msg.str());
    }
  }
  auto field = args["field"].as<std::string>();
  auto output_dir = args["output-dir"].as<std::string>();
  Run(camera, angle, resize, x_dim, y_dim, field, output_dir);
  return 0;
}
