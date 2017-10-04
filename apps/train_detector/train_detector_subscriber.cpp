// This application attaches to a published frame stream and stores the frames
// on disk.

#include <atomic>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>

#include "common/context.h"
#include "common/types.h"
#include "processor/compressor.h"
#include "processor/frame_writer.h"
#include "processor/jpeg_writer.h"
#include "processor/pubsub/frame_subscriber.h"
#include "stream/stream.h"

namespace po = boost::program_options;

std::atomic<bool> stopped(false);

void ProgressTracker(StreamPtr stream) {
  StreamReader* reader = stream->Subscribe();
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      std::cout << "\rReceived frame "
                << frame->GetValue<unsigned long>("frame_id") << " with time "
                << frame->GetValue<boost::posix_time::ptime>(
                       "capture_time_micros");
      // This is required in order to make the console update as soon as the
      // above log is printed. Without this, the progress log will not update
      // smoothly.
      std::cout.flush();
    }
  }
  reader->UnSubscribe();
}

void Run(const std::string& publisher_url, bool compress,
         std::unordered_set<std::string> fields_to_save,
         bool save_fields_separately, const std::string& output_dir) {
  LOG(INFO) << "Detecting trains...";

  std::vector<std::shared_ptr<Processor>> procs;

  // Create FrameSubscriber.
  auto subscriber = std::make_shared<FrameSubscriber>(publisher_url);
  procs.push_back(subscriber);
  StreamPtr network_stream = subscriber->GetSink();

  StreamPtr stream_to_write;
  if (compress) {
    // Create Compressor.
    auto compressor =
        std::make_shared<Compressor>(Compressor::CompressionType::BZIP2);
    compressor->SetSource(network_stream);
    procs.push_back(compressor);
    stream_to_write = compressor->GetSink();
  } else {
    stream_to_write = network_stream;
  }

  // Create FrameWriter.
  auto frame_writer = std::make_shared<FrameWriter>(
      fields_to_save, output_dir, FrameWriter::FileFormat::BINARY,
      save_fields_separately, true);
  frame_writer->SetSource(stream_to_write);
  procs.push_back(frame_writer);

  // Create JpegWriter.
  auto jpeg_writer = std::make_shared<JpegWriter>("original_image", output_dir);
  jpeg_writer->SetSource(stream_to_write);
  procs.push_back(jpeg_writer);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  std::thread progress_thread =
      std::thread([stream_to_write] { ProgressTracker(stream_to_write); });

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }

  // Signal the progress thread to stop.
  stopped = true;
  progress_thread.join();
}

int main(int argc, char* argv[]) {
  po::options_description desc("Train Detector");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's config files.");
  desc.add_options()("publisher-url,u", po::value<std::string>()->required(),
                     "The URL (host:port) on which frames are being "
                     "published.");
  desc.add_options()("compress,c",
                     "Whether to compress the \"original_bytes\" field.");
  desc.add_options()(
      "fields-to-save",
      po::value<std::vector<std::string>>()
          ->multitoken()
          ->composing()
          ->default_value(std::vector<std::string>{"original_bytes"},
                          "{original_bytes}"),
      "The fields to save.");
  desc.add_options()("save-fields-separately",
                     "Whether to save each frame field in a separate file.");
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
  bool compress = args.count("compress");
  bool save_fields_separately = args.count("save-fields-separately");
  std::vector<std::string> fields_to_save =
      args["fields-to-save"].as<std::vector<std::string>>();
  std::string output_dir = args["output-dir"].as<std::string>();
  Run(publisher_url, compress,
      std::unordered_set<std::string>{fields_to_save.begin(),
                                      fields_to_save.end()},
      save_fields_separately, output_dir);
  return 0;
}
