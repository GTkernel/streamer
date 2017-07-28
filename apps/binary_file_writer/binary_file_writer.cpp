// The binary_file_writer is a simple app that reads frames from a single camera
// and immediately saves their raw pixel data to disk.

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "processor/binary_file_writer.h"
#include "processor/processor.h"

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

void Run(const std::string& camera_name, const std::string& output_dir) {
  // Create Camera.
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Create BinaryFileWriter.
  auto writer =
      std::make_shared<BinaryFileWriter>("original_bytes", output_dir);
  writer->SetSource(camera->GetStream());
  procs.push_back(writer);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  while (true) continue;
}

int main(int argc, char* argv[]) {
  po::options_description desc("Stores raw frame data.");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to store the raw frame data.");

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
  std::string camera = args["camera"].as<std::string>();
  std::string output_dir = args["output-dir"].as<std::string>();
  Run(camera, output_dir);
  return 0;
}
