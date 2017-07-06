/**
 * @brief multicam.cpp - An example showing the usage to run realtime
 * classification on multiple camera streams. This example reads frames from
 * multiple cameras, overlays labels with each camera input, filter
 * `unimportant' videos and store the video and classification results locally.
 */

#include <boost/program_options.hpp>
#include <csignal>
#include "streamer.h"

#include "db_filewriter.h"
#include "compressor.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

/////// Global vars
std::vector<std::shared_ptr<Camera>> cameras;

void CleanUp() {
  for (const auto& camera : cameras) {
    if (camera->IsStarted()) camera->Stop();
  }
}

void SignalHandler(int) {
  std::cout << "Received SIGINT, try to gracefully exit" << std::endl;
  //  CleanUp();

  exit(0);
}

void Run(const std::vector<string>& camera_names, std::string root_dir) {
  cout << "Detect Trains" << endl;

  std::signal(SIGINT, SignalHandler);

  CameraManager& camera_manager = CameraManager::GetInstance();

  // Check options
  for (const auto& camera_name : camera_names) {
    CHECK(camera_manager.HasCamera(camera_name))
        << "Camera " << camera_name << " does not exist";
  }

  ////// Start cameras, processors
  for (const auto& camera_name : camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  auto* compressor = new Compressor(Compressor::CompressionType::BZIP2);
  compressor->SetSource("input", cameras[0]->GetSink("output"));
  // Do video stream classification
  auto* db_fw = new DBFileWriter(root_dir);
  db_fw->SetSource("input", compressor->GetSink("output"));

  compressor->Start();
  db_fw->Start();

  for (const auto& camera : cameras) {
    if (!camera->IsStarted()) {
      camera->Start();
    }
  }

  while (true)
    ;

  LOG(INFO) << "Done";
}

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Trains");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera,c",
                     po::value<string>()->value_name("CAMERAS")->required(),
                     "The name of the camera to use, if there are multiple "
                     "cameras to be used, separate with ,");
  desc.add_options()("device", po::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()(
      "rootdir", po::value<string>()->value_name("ROOTDIR")->required(),
      "Root directory for the directory tree containing all of the images");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");

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

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();
  int device_number = vm["device"].as<int>();
  std::string root_dir = vm["rootdir"].as<std::string>();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);

  auto camera_names = SplitString(vm["camera"].as<string>(), ",");
  Run(camera_names, root_dir);

  return 0;
}
