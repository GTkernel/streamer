// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This application throttles a camera stream and publishes it on the network.

#include <atomic>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "processor/pubsub/frame_publisher.h"
#include "processor/throttler.h"
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
      std::cout << "\rSent frame " << frame->GetValue<unsigned long>("frame_id")
                << " from time: "
                << frame->GetValue<boost::posix_time::ptime>(
                       Camera::kCaptureTimeMicrosKey);
      // This is required in order to make the console update as soon as the
      // above log is printed. Without this, the progress log will not update
      // smoothly.
      std::cout.flush();
    }
  }
  reader->UnSubscribe();
}

void Run(const std::string& camera_name, double fps,
         std::unordered_set<std::string> fields_to_send,
         const std::string& publish_url) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  StreamPtr stream = camera->GetStream();
  if (fps) {
    // Create Throttler.
    auto throttler = std::make_shared<Throttler>(fps);
    throttler->SetSource(stream);
    procs.push_back(throttler);
    stream = throttler->GetSink();
  }

  // Create FramePublisher.
  auto publisher =
      std::make_shared<FramePublisher>(publish_url, fields_to_send);
  publisher->SetSource(stream);
  procs.push_back(publisher);

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
  std::vector<std::string> default_fields = {"frame_id",
                                             Camera::kCaptureTimeMicrosKey,
                                             "start_time_ms", "original_image"};
  std::ostringstream default_fields_str;
  default_fields_str << "{ ";
  for (const auto& field : default_fields) {
    default_fields_str << "\"" << field << "\" ";
  }
  default_fields_str << "}";

  po::options_description desc("Publishes a frame stream on the network");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("fps,f", po::value<double>()->default_value(0),
                     ("The desired maximum rate of the published stream. The "
                      "actual rate may be less. An fps of 0 disables "
                      "throttling."));
  desc.add_options()(
      "fields-to-send",
      po::value<std::vector<std::string>>()
          ->multitoken()
          ->composing()
          ->default_value(default_fields, default_fields_str.str()),
      "The fields to publish.");
  desc.add_options()(
      "publish-url,u",
      po::value<std::string>()->default_value("127.0.0.1:5536"),
      "The URL (host:port) on which to publish the frame stream.");

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
  auto fields_to_send = args["fields-to-send"].as<std::vector<std::string>>();
  auto publish_url = args["publish-url"].as<std::string>();
  Run(camera_name, fps,
      std::unordered_set<std::string>{fields_to_send.begin(),
                                      fields_to_send.end()},
      publish_url);
  return 0;
}
