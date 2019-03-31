#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "camera/camera_manager.h"
#include "processor/image_transformer.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/neural_net_evaluator.h"
#include "processor/rpc/frame_sender.h"
#include "processor/processor.h"

namespace po = boost::program_options;

void Run(const std::string& publish_endpoint, const std::string& camera_name,
         const std::string& net, const std::string& input_layer,
         const std::string& output_layer,
         const int exec_sec ) {

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);

  // Transformer
  auto model_desc = ModelManager::GetInstance().GetModelDesc(net);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());

  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetStream());

  // NNE
  std::vector<std::string> output_layers = {output_layer};
  auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1, output_layers);
  nne->SetSource(transformer->GetSink(), input_layer);

  auto sender = new FrameSender(publish_endpoint, {"frame_id", "start_time_ms", "before_nne_time", output_layer});
  sender->SetSource(nne->GetSink());

  auto reader = sender->GetSink()->Subscribe();
  int frame_count = 0;
  
  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();

  sender->Start();
  nne->Start();
  transformer->Start();
  camera->Start();
 
  while (true) {
    auto frame = reader->PopFrame(20);
    if(frame != NULL) {
        frame_count ++;
    }
    auto passing_time = boost::posix_time::microsec_clock::local_time() - processing_start_micros_;
    if ( passing_time.total_seconds() > exec_sec){
        break;
    }
  }


  sender->Stop();
  camera->Stop();
  transformer->Stop();
  nne->Stop();

  // Stop the processors in forward order.

  std::cout << "======" << std::endl;
  std::cout << "frame count = " << frame_count << std::endl;
  std::cout << "nne fps = " << nne->GetHistoricalProcessFps() << std::endl;
  std::cout << "nne latency = " << nne->GetAvgProcessingLatencyMs() << std::endl;
  std::cout << "nne queue = " << nne->GetAvgQueueLatencyMs() << std::endl;
  std::cout << "sender fps = " << sender->GetHistoricalProcessFps() << std::endl;
  std::cout << "sender latency = " << sender->GetAvgProcessingLatencyMs() << std::endl;
//  std::cout << "serialize latency = " << sender->serialize_latency_ms_sum / frame_count << std::endl;
//  std::cout << "real sending latency = " << sender->avg_send_latency_ms << std::endl;
//  std::cout << "sender queue = " << sender->GetAvgQueueLatencyMs() << std::endl;
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
  desc.add_options()("input,i", po::value<std::string>()->required(),
                     "The name of the input layer of the neural net.");
  desc.add_options()("output,o", po::value<std::string>()->required(),
                     "The name of the output layer of the neural net.");
  desc.add_options()("publish_url,p",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "host:port to publish (e.g., example.com:4444)"); 
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
    std::string input = args["input"].as<std::string>();
    std::string output = args["output"].as<std::string>();
    int exec_sec = args["execution_time"].as<int>();

    Run(publish_endpoint, camera_name, net, input, output, exec_sec);
    return 0;
}


