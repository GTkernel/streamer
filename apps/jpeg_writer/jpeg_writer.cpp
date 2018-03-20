// The JpegWriter is a simple app that reads frames from a single camera and
// immediately saves them as JPEG images.

#include <cstdio>
#include <iostream>
#include <memory>
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
#include "processor/jpeg_writer.h"
#include "processor/processor.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, bool resize, int x_dim, int y_dim,
         const std::string& field, const std::string& output_dir,
         bool organize_by_time, unsigned long frames_per_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  StreamPtr stream = camera->GetStream();
  std::string field_to_save;
  if (resize) {
    // Create ImageTransformer.
    auto transformer =
        std::make_shared<ImageTransformer>(Shape(3, x_dim, y_dim), true);
    transformer->SetSource(camera->GetStream());
    procs.push_back(transformer);
    stream = transformer->GetSink();

    // The ImageTransformer is hardcorded to store the resized image at the key
    // "image".
    field_to_save = "image";
  } else {
    field_to_save = field;
  }

  // Create JpegWriter.
  auto writer = std::make_shared<JpegWriter>(field_to_save, output_dir,
                                             organize_by_time, frames_per_dir);
  writer->SetSource(stream);
  procs.push_back(writer);

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
  desc.add_options()("x-dim,x", po::value<int>(),
                     "The width to which to resize the frames. Forces "
                     "\"--field\" to be \"original_image\".");
  desc.add_options()("y-dim,y", po::value<int>(),
                     "The height to which to resize the frames. Forces "
                     "\"--field\" to be \"original_image\".");
  desc.add_options()("field,f",
                     po::value<std::string>()->default_value("original_image"),
                     "The field to save as a JPEG. Assumed to be "
                     "\"original_image\" when using \"--resize\".");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to store the frame JPEGs.");
  desc.add_options()("organize-by-time,t",
                     "Whether to organize the output file by date and time.");
  desc.add_options()("frames-per-dir,n",
                     po::value<unsigned long>()->default_value(1000),
                     "The number of frames to save in each subdir. Only valid "
                     "if \"--organize-by-time\" is not specified.");

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

  auto camera = args["camera"].as<std::string>();
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
  auto field = args["field"].as<std::string>();
  auto output_dir = args["output-dir"].as<std::string>();
  bool organize_by_time = args.count("organize-by-time");
  auto frames_per_dir = args["frames-per-dir"].as<unsigned long>();
  Run(camera, resize, x_dim, y_dim, field, output_dir, organize_by_time,
      frames_per_dir);
  return 0;
}
