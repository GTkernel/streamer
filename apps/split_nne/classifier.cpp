#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "common/context.h"
#include "model/model_manager.h"
#include "processor/image_classifier.h"
#include "processor/processor.h"
#include "processor/pubsub/frame_subscriber.h"

#include "tensorflow/core/framework/tensor.h"

namespace po = boost::program_options;

void Run(const std::string& subscribe_endpoint, const std::string& net,
         const int exec_sec){

  auto subscriber = new FrameSubscriber(subscribe_endpoint);
  subscriber->SetBlockOnPush(true);
    
  auto model_desc = ModelManager::GetInstance().GetModelDesc(net);

  //classifier (NNC)
  auto classifier = std::make_shared<ImageClassifier>(model_desc, 1);
  classifier->SetSource("input", subscriber->GetSink());

  boost::posix_time::time_duration total_nne1_micros, total_nne2_micros, total_nne1_eval, total_nne2_eval, 
                                   total_nne1_preproc_micros, total_nne2_preproc_micros,
                                   total_nne1_push_micros, total_nne2_push_micros;
  double push_rate = 0;
  int frame_count = 0;
  tensorflow::Tensor split_tensor;

  classifier->Start();
  subscriber->Start();

  auto reader = classifier->GetSink("output")->Subscribe();
  
  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();
  while (true) {
    auto frame = reader->PopFrame();
    if (frame->IsStopFrame()) break;
    // Get NNE excution time
    total_nne1_micros += frame->GetValue<boost::posix_time::time_duration>("NeuralNetEvaluator.total_micros_1");
    total_nne2_micros += frame->GetValue<boost::posix_time::time_duration>("NeuralNetEvaluator.total_micros_2");
    total_nne1_eval += frame->GetValue<boost::posix_time::time_duration>("eval_micros_1");
    total_nne2_eval +=  frame->GetValue<boost::posix_time::time_duration>("eval_micros_2");
    total_nne1_preproc_micros += frame->GetValue<boost::posix_time::time_duration>("preproc_micros_1");
    total_nne2_preproc_micros += frame->GetValue<boost::posix_time::time_duration>("preproc_micros_2");

    frame_count += 1;
    push_rate += reader->GetPushFps();

    auto passing_time = boost::posix_time::microsec_clock::local_time() - processing_start_micros_;
    if (passing_time.total_seconds() > exec_sec) break;
  }

  // Stop the processors in forward order.
  subscriber->Stop();
  classifier->Stop();

  std::cout << "======" << std::endl;
  std::cout << "Frame count = " << frame_count << std::endl;
  std::cout << "Average NNE1 time = " << total_nne1_micros.total_microseconds() / frame_count << std::endl;
  std::cout << "Average NNE2 time = " << total_nne2_micros.total_microseconds() / frame_count << std::endl;

  std::cout << "Average NNE1 eval time = " << total_nne1_eval.total_microseconds() / frame_count << std::endl;
  std::cout << "Average NNE2 eval time = " << total_nne2_eval.total_microseconds() / frame_count << std::endl;

  std::cout << "Average NNE1 pproc time = " << total_nne1_preproc_micros.total_microseconds() / frame_count << std::endl;
  std::cout << "Average NNE2 pproc time = " << total_nne2_preproc_micros.total_microseconds() / frame_count << std::endl;
  std::cout << "Push rate (frame/s) = " << push_rate / frame_count<< std::endl;

}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Demonstrates splitting DNN evaluation across two NNEs");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("net,n", po::value<std::string>()->required(),
                     "The name of the neural net to run.");
  desc.add_options()("subscribe_url,s",
                     po::value<std::string>()->default_value("127.0.0.1:5538"),
                     "host:port to connect (e.g., example.com:4444)");
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
  std::string subscribe_endpoint = args["subscribe_url"].as<std::string>();
  std::string net = args["net"].as<std::string>();
  int exec_sec = args["execution_time"].as<int>();

  Run(subscribe_endpoint, net, exec_sec);
  return 0;
}
