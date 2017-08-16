// This is a sample example app that uses the strider processor
//   Camera -> Strider

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "processor/processor.h"
#include "processor/strider.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, unsigned int stride) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Strider
  auto strider = std::make_shared<Strider>(stride);
  strider->SetSource(camera->GetStream());
  procs.push_back(strider);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  std::cout << "Press \"Control-C\" to stop." << std::endl;
  auto reader = strider->GetSink("output")->Subscribe();
  while (true) {
    auto frame = reader->PopFrame();
    LOG(INFO) << "Received frame: "
              << frame->GetValue<unsigned long>("frame_id");
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Simple camera display test");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("stride,s", po::value<int>()->default_value(50),
                     "The stride of the pipeline.");

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
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  std::string camera_name = args["camera"].as<std::string>();
  int stride = args["stride"].as<int>();
  if (stride < 0) {
    std::cerr << "\"--stride\" cannot be negative, but is: " << stride
              << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  Run(camera_name, (unsigned int)stride);
  return 0;
}
