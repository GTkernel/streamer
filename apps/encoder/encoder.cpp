/**
 * @brief encoder.cpp - An example application showing the usage of encoder.
 */

#include <cstdio>

#include <boost/program_options.hpp>

#include "streamer.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

void Run(const std::string& camera_name, std::string& dst_file, int port) {
  if (dst_file == "" && port == -1) {
    cout << "Specify output_filename or port" << endl;
    return;
  }

  CameraManager& camera_manager = CameraManager::GetInstance();

  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  auto camera = camera_manager.GetCamera(camera_name);
  auto camera_stream = camera->GetStream();

  LOG(INFO) << "Camera image size: " << camera->GetWidth() << "x"
            << camera->GetHeight();

  // Encoder
  std::shared_ptr<Processor> encoder;
  if (dst_file != "") {
    cout << "Store video to: " << dst_file << endl;
    encoder = std::shared_ptr<Processor>(
        new GstVideoEncoder("original_image", dst_file));
    encoder->SetSource("input", camera_stream);
  } else if (port != -1) {
    cout << "Stream video on port: " << port << endl;
    encoder = std::shared_ptr<Processor>(
        new GstVideoEncoder("original_image", port, false));
    encoder->SetSource("input", camera_stream);
    // Receive pipeline
    // gst-launch-1.0 -v udpsrc port=5000 ! application/x-rtp ! rtph264depay !
    // avdec_h264 ! videoconvert ! autovideosink sync=false
    //
  }

  camera->Start();
  encoder->Start();

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  encoder->Stop();
  camera->Stop();
}

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("GigE camera demo");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera",
                     po::value<std::string>()->value_name("CAMERA")->required(),
                     "The name of the camera to use");
  desc.add_options()("output,o", po::value<std::string>()->value_name("OUTPUT"),
                     "The name of the file to store the video file");
  desc.add_options()("port,p", po::value<int>()->value_name("PORT"),
                     "The port to be used to stream the video");
  desc.add_options()("config_dir,C",
                     po::value<std::string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << endl;
    cout << desc << endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<std::string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<std::string>();
  std::string output_filename;
  int port = -1;
  if (vm.count("output") != 0) output_filename = vm["output"].as<std::string>();
  if (vm.count("port") != 0) port = vm["port"].as<int>();

  Run(camera_name, output_filename, port);

  return 0;
}
