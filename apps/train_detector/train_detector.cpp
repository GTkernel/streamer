// This application stores frames that contain trains.

#include <atomic>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "processor/compressor.h"
#include "processor/frame_writer.h"
#include "processor/jpeg_writer.h"
#include "processor/throttler.h"
#include "processor/train_detector.h"
#include "stream/frame.h"
#include "stream/stream.h"

namespace po = boost::program_options;

// Whether the pipeline has been stopped.
std::atomic<bool> stopped(false);

void ProgressTracker(StreamPtr stream) {
  StreamReader* reader = stream->Subscribe();
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      std::cout << "\rReceived frame "
                << frame->GetValue<unsigned long>("frame_id") << " from time: "
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

void Run(const std::string& camera_name, double fps,
         unsigned long num_leading_frames, unsigned long num_trailing_frames,
         std::unordered_set<std::string> fields_to_save,
         bool save_fields_separately, bool save_original_bytes, bool compress,
         bool save_jpegs, const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  StreamPtr camera_stream = camera->GetStream();
  StreamPtr stream = camera_stream;
  if (fps != 0) {
    // Create Throttler to decrease camera FPS to desired (high) FPS, in case
    // the camera is capturing faster than the train detection algorithm can
    // handle).
    auto throttler = std::make_shared<Throttler>(fps);
    throttler->SetSource(stream);
    procs.push_back(throttler);
    stream = throttler->GetSink();
  }

  // Create TrainDetector. Connect to Throttler.
  auto detector =
      std::make_shared<TrainDetector>(num_leading_frames, num_trailing_frames);
  detector->SetSource(stream);
  procs.push_back(detector);

  if (!fields_to_save.empty()) {
    // Create FrameWriter for frame fields (i.e., metadata).
    auto frame_writer = std::make_shared<FrameWriter>(
        fields_to_save, output_dir, FrameWriter::FileFormat::JSON,
        save_fields_separately, true);
    frame_writer->SetSource(stream);
    procs.push_back(frame_writer);
  }

  if (save_original_bytes) {
    if (compress) {
      // Create Compressor.
      auto compressor =
          std::make_shared<Compressor>(Compressor::CompressionType::BZIP2);
      compressor->SetSource(stream);
      procs.push_back(compressor);
      stream = compressor->GetSink();
    }

    // Create FrameWriter for writing original demosaiced image.
    auto image_writer = std::make_shared<FrameWriter>(
        std::unordered_set<std::string>{"original_image"}, output_dir,
        FrameWriter::FileFormat::BINARY, save_fields_separately, true);
    image_writer->SetSource(stream);
    procs.push_back(image_writer);
  }

  if (save_jpegs) {
    // Create JpegWriter.
    auto jpeg_writer =
        std::make_shared<JpegWriter>("original_image", output_dir, true);
    jpeg_writer->SetSource(stream);
    procs.push_back(jpeg_writer);
  }

  std::thread progress_thread =
      std::thread([camera_stream] { ProgressTracker(camera_stream); });

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

  // Signal the progress thread to stop.
  stopped = true;
  progress_thread.join();
}

int main(int argc, char* argv[]) {
  po::options_description desc("Saves frames containing trains");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("fps,f", po::value<double>()->default_value(0),
                     ("The desired maximum rate of the published stream. The "
                      "actual rate may be less. An fps of 0 disables "
                      "throttling."));
  desc.add_options()("num-leading-frames,l",
                     po::value<unsigned long>()->default_value(100),
                     "The number of frames to save from before each train "
                     "appears.");
  desc.add_options()("num-trailing-frames,l",
                     po::value<unsigned long>()->default_value(100),
                     "The number of frames to save from after each train "
                     "appears.");
  desc.add_options()("fields-to-save",
                     po::value<std::vector<std::string>>()
                         ->multitoken()
                         ->composing()
                         ->default_value({}, "None"),
                     "The fields to save.");
  desc.add_options()("save-fields-separately",
                     "Whether to save each frame field in a separate file.");
  desc.add_options()("save-original-bytes",
                     "Whether to save the uncompressed, demosaiced image.");
  desc.add_options()("compress,c",
                     "Whether to compress the \"original_bytes\" field.");
  desc.add_options()("save-jpegs",
                     "Whether to save a JPEG of the "
                     "\"original_image\" field.");
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

  auto camera_name = args["camera"].as<std::string>();
  auto fps = args["fps"].as<double>();
  auto num_leading_frames = args["num-leading-frames"].as<unsigned long>();
  auto num_trailing_frames = args["num-trailing-frames"].as<unsigned long>();
  auto fields_to_save = args["fields-to-save"].as<std::vector<std::string>>();
  bool save_fields_separately = args.count("save-fields-separately");
  bool save_original_bytes = args.count("save-original-bytes");
  bool compress = args.count("compress");
  bool save_jpegs = args.count("save-jpegs");
  auto output_dir = args["output-dir"].as<std::string>();

  Run(camera_name, fps, num_leading_frames, num_trailing_frames,
      std::unordered_set<std::string>{fields_to_save.begin(),
                                      fields_to_save.end()},
      save_fields_separately, save_original_bytes, compress, save_jpegs,
      output_dir);
  return 0;
}
