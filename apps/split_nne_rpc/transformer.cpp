#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "camera/camera_manager.h"
#include "model/model_manager.h"
#include "processor/image_transformer.h"
#include "common/context.h"
#include "processor/processor.h"
#include "processor/rpc/frame_sender.h"

namespace po = boost::program_options;

void Run(const std::string& publish_endpoint, const std::string& camera_name, 
         const std::string& net, const int exec_sec){

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);

  auto model_desc = ModelManager::GetInstance().GetModelDesc(net);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());

  // Transformer
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetStream());

  auto sender = new FrameSender(publish_endpoint, {"frame_id", "start_time_ms", "before_nne_time", "image"});
  sender->SetSource(transformer->GetSink());

  auto reader = sender->GetSink()->Subscribe();
  int frame_count = 0; 
  int frame_id = 0;

  sender->Start();
  transformer->Start();
  camera->Start();

  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();
  
  while (true) {
   auto frame = reader->PopFrame(20);

   if (frame != NULL) {
     frame_id = frame->GetValue<unsigned long>("frame_id"); 
     frame_count ++;
   }

   auto passing_time = boost::posix_time::microsec_clock::local_time() - processing_start_micros_;
   if (passing_time.total_seconds() > exec_sec || frame_id >= 2519 ) break;
  }

  // Stop the processors in forward order.
  sender->Stop();
  transformer->Stop();
  camera->Stop();

  std::cout << "======" << std::endl;
  std::cout << "frame count = " << frame_count << std::endl;
  std::cout << "camera fps = " << camera->GetHistoricalProcessFps() << std::endl;
  std::cout << "camera latency = " << camera->GetAvgProcessingLatencyMs() << std::endl;
  std::cout << "transformer fps = " << transformer->GetHistoricalProcessFps() << std::endl;
  std::cout << "transformer latency = " << transformer->GetAvgProcessingLatencyMs() << std::endl;
  std::cout << "transformer queue = " << transformer->GetAvgQueueLatencyMs() << std::endl;
  std::cout << "sender fps = " << sender->GetHistoricalProcessFps() << std::endl;
  std::cout << "sender latency = " << sender->GetAvgProcessingLatencyMs() << std::endl;
//  std::cout << "serialize latency = " << sender->serialize_latency_ms_sum / frame_count << std::endl;
//  std::cout << "sender queue = " << sender->GetAvgQueueLatencyMs() << std::endl;
  std::cout << "real sending latency = " << sender->avg_send_latency_ms << std::endl;
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
  desc.add_options()("publish_url,p",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "host:port to connect to (e.g., example.com:4444)");
  desc.add_options()("net,n", po::value<std::string>()->required(),
                     "The name of the neural net to run.");
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
  std::string publish_endpoint = args["publish_url"].as<std::string>();
  std::string net = args["net"].as<std::string>();
  int exec_sec = args["execution_time"].as<int>();
 
  Run(publish_endpoint, camera_name, net, exec_sec);
  return 0;
}
