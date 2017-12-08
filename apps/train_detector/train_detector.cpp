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

void Run(const std::string& camera_name) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Create Throttler (decrease camera FPS to desired (high) FPS, in case the
  // camera is capturing faster than the train detection algorithm can handle).
  // Connect to Camera.

  // Create TrainDetector. Connect to Throttler.

  // Create Buffer. Connect to Throttler.

  // Create FrameWriter. Connect to Buffer.

  StreamPtr stream = camera->GetStream();
  std::thread progress_thread =
      std::thread([stream] { ProgressTracker(stream); });

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

  std::string camera_name = args["camera"].as<std::string>();
  Run(camera_name);
  return 0;
}
