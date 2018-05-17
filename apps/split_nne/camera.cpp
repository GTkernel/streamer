#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/image_transformer.h"
#include "processor/processor.h"
#include "processor/pubsub/frame_publisher.h"

namespace po = boost::program_options;

void Run(const std::string& publish_endpoint, const std::string& camera_name, 
         const std::string& net, const int exec_sec){

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);

  // Transformer
  auto model_desc = ModelManager::GetInstance().GetModelDesc(net);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetStream());
  transformer->SetBlockOnPush(true);

  auto publisher = new FramePublisher(publish_endpoint, {"image", "frame_id", "start_time_ms"});
  publisher->SetSource(transformer->GetSink("output"));

  publisher->Start();
  transformer->Start();
  camera->Start();

  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();
 
  while (true) {
   auto passing_time = boost::posix_time::microsec_clock::local_time() - processing_start_micros_;
   if (passing_time.total_seconds() > exec_sec) break;
  }

  // Stop the processors in forward order.
  camera->Stop();
  transformer->Stop();
  publisher->Stop();
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Demonstrates splitting DNN evaluation across two NNEs");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("net,n", po::value<std::string>()->required(),
                     "The name of the neural net to run.");
  desc.add_options()("publish_url,p",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "host:port to connect to (e.g., example.com:4444)");
  desc.add_options()("execution_time,t",po::value<int>()->default_value(120),
                     "the time for running the application(seconds)");

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
  std::string net = args["net"].as<std::string>();
  std::string publish_endpoint = args["publish_url"].as<std::string>();
  int exec_sec = args["execution_time"].as<int>();
 
  Run(publish_endpoint, camera_name, net, exec_sec);
  return 0;
}
