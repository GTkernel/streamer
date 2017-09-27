// This application attaches to a published frame stream and stores the raw
// image data, a metadata JSON file, and a JPEG thumbnail image on disk.

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "common/context.h"
#include "db_filewriter.h"
#include "processor/compressor.h"
#include "processor/pubsub/frame_subscriber.h"

namespace po = boost::program_options;

void Run(const std::string& publisher_url, const std::string& output_dir) {
  LOG(INFO) << "Detecting trains...";

  std::vector<std::shared_ptr<Processor>> procs;

  // Create FrameSubscriber.
  auto subscriber = std::make_shared<FrameSubscriber>(publisher_url);
  procs.push_back(subscriber);

  // Create Compressor.
  auto compressor =
      std::make_shared<Compressor>(Compressor::CompressionType::BZIP2);
  compressor->SetSource(subscriber->GetSink());
  procs.push_back(compressor);

  // Create DBFileWriter.
  auto writer = std::make_shared<DBFileWriter>(output_dir);
  writer->SetSource("input", compressor->GetSink());
  procs.push_back(writer);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Train Detector");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("publisher-url,u", po::value<std::string>()->required(),
                     "The URL (host:port) on which frames are being "
                     "published.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     ("The root directory of the image storage database."));

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

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  std::string publisher_url = args["publisher-url"].as<std::string>();
  std::string output_dir = args["output-dir"].as<std::string>();
  Run(publisher_url, output_dir);
  return 0;
}
