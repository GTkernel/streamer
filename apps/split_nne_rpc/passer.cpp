#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "common/context.h"
#include "processor/rpc/frame_sender.h"
#include "processor/rpc/frame_receiver.h"
#include "processor/processor.h"

namespace po = boost::program_options;

void Run(const std::string& subscribe_endpoint,
         const std::string& publish_endpoint,
         const int exec_sec ) {


  auto receiver = new FrameReceiver(subscribe_endpoint);

  auto sender = new FrameSender(publish_endpoint);
  sender->SetSource(receiver->GetSink());

  auto reader = sender->GetSink()->Subscribe();
  int frame_count = 0;
  
  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();

  receiver->Start();
  sender->Start();

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

  receiver->Stop();
  sender->Stop();
  // Stop the processors in forward order.

  std::cout << "======" << std::endl;
  std::cout << "frame count = " << frame_count << std::endl;
  std::cout << "sender latency = " << sender->GetAvgProcessingLatencyMs() << std::endl;
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Demonstrates splitting DNN evaluation across two NNEs");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("publish_url,p",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
                     "host:port to publish (e.g., example.com:4444)"); 
  desc.add_options()("subscribe_url,s",
                     po::value<std::string>()->default_value("127.0.0.1:5536"),
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
    std::string publish_endpoint = args["publish_url"].as<std::string>();
    int exec_sec = args["execution_time"].as<int>();

    Run(subscribe_endpoint, publish_endpoint, exec_sec);
    return 0;
}


