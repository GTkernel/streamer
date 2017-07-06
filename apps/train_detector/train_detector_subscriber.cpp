/**
 * @brief train_detector_subscriber.cpp - This application attaches to a
 * published frame stream, compresses the raw frame data, and stores it on disk.
 */

#include <csignal>

#include <boost/program_options.hpp>

#include "db_filewriter.h"
#include "processor/pubsub/frame_subscriber.h"
#include "streamer.h"

namespace po = boost::program_options;

std::shared_ptr<FrameSubscriber> subscriber;
std::shared_ptr<DBFileWriter> writer;

void CleanUp() {
  if (subscriber != nullptr && subscriber->IsStarted()) subscriber->Stop();
  if (writer != nullptr && writer->IsStarted()) writer->Stop();
}

void SignalHandler(int) {
  LOG(INFO) << "Received SIGINT, trying to exit gracefully...";
  CleanUp();
  exit(0);
}

void Run(const std::string& publisher_url, const std::string& output_dir) {
  LOG(INFO) << "Detecting trains...";

  subscriber = std::make_shared<FrameSubscriber>(publisher_url);

  writer = std::make_shared<DBFileWriter>(output_dir);
  writer->SetSource("input", subscriber->GetSink());

  writer->Start();
  subscriber->Start();

  while (true) {
  }
}

int main(int argc, char* argv[]) {
  std::signal(SIGINT, SignalHandler);

  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Train Detector");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("publisher_url,u", po::value<std::string>()->required(),
                     "The URL on which frames are being published.");
  desc.add_options()("output_dir,o", po::value<std::string>()->required(),
                     ("The root directory of the of the image storage "
                      "database."));

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  // Parse arguments
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<std::string>());
  }
  auto publisher_url = vm["publisher_url"].as<std::string>();
  auto output_dir = vm["output_dir"].as<std::string>();

  // Init streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  Run(publisher_url, output_dir);
  return 0;
}
