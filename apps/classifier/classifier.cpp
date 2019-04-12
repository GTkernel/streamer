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

// The classifier app demonstrates how to use an ImageClassifier processor.

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/image_classifier.h"
#include "processor/image_transformer.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, const std::string& model_name,
         bool display, const int exec_sec ) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // ImageTransformer
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model_name);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetSink("output"));
  procs.push_back(transformer);

  // ImageClassifier
  auto classifier =
      std::make_shared<ImageClassifier>(model_desc, input_shape, 1);
  classifier->SetSource("input", transformer->GetSink("output"));
  procs.push_back(classifier);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  if (display) {
    std::cout << "Press \"q\" to stop." << std::endl;
  } else {
    std::cout << "Press \"Control-C\" to stop." << std::endl;
  }

  auto reader = classifier->GetSink("output")->Subscribe();
  int frame_count = 0;
  int frame_id = 0;
  boost::posix_time::time_duration total_nne_eval;
  double classifier_eval_time;

  auto processing_start_micros_ = boost::posix_time::microsec_clock::local_time();

  while (true) {
    auto frame = reader->PopFrame(20);
    if (frame != NULL) {
        frame_id = frame->GetValue<unsigned long>("frame_id");
        total_nne_eval += frame->GetValue<boost::posix_time::time_duration>("eval_micros");
        frame_count++;
    }
    auto passing_time = boost::posix_time::microsec_clock::local_time() - processing_start_micros_;
    if ( passing_time.total_seconds() > exec_sec ){
        classifier_eval_time = classifier->GetAvgProcessingLatencyMs();
        break;
    }
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
  
  std::cout << "======" << std::endl;
  std::cout << "Frame count = " << frame_count << std::endl;
  std::cout << "camera = " << camera->GetAvgProcessingLatencyMs() << std::endl;
  std::cout << "transformer = " << transformer->GetAvgProcessingLatencyMs() << std::endl;
  std::cout << "nne = " << total_nne_eval.total_microseconds() / frame_count << std::endl;
  std::cout << "classifier = " << classifier_eval_time << std::endl;

}

int main(int argc, char* argv[]) {
  po::options_description desc("Runs image classification on a video stream");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("display,d", "Enable display or not");
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
  auto camera_name = args["camera"].as<std::string>();
  auto model = args["model"].as<std::string>();
  int exec_sec = args["execution_time"].as<int>();
  bool display = args.count("display");
  Run(camera_name, model, display, exec_sec);
  return 0;
}
