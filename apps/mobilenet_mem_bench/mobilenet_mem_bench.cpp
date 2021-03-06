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

// This app allows you to profile the memory usage and throughput of different
// models.

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

#include "processor/neuralnet_bench.h"
#include "utils/perf_utils.h"

namespace po = boost::program_options;

int num_classifiers;
int batch_size;
bool do_mem;

void Run(const std::string& camera_name, const std::string& model_name) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // ImageTransformer
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model_name);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource(camera->GetStream());
  procs.push_back(transformer);

  std::shared_ptr<NNBench> nn_bench = std::make_shared<NNBench>(
      model_desc, input_shape, batch_size, num_classifiers);
  procs.push_back(nn_bench);
  nn_bench->SetSource(transformer->GetSink());

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  auto reader = nn_bench->GetSink("output")->Subscribe();
  int destroy_counter = 0;
  std::cout << "Num Classifiers"
            << ","
            << "Virtual Mem (kb)"
            << ","
            << "Physical Mem (kb)"
            << ","
            << "Runtime: " << std::endl;
  while (true) {
    auto frame = reader->PopFrame();

    long nnbench_micros = frame->GetValue<long>("neuralnet_bench.micros");
    std::cout << num_classifiers << "," << GetVirtualKB() << ","
              << GetPhysicalKB() << "," << nnbench_micros << std::endl;
    destroy_counter += 1;
    if (destroy_counter == 500) {
      std::terminate();
    }
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
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
  desc.add_options()("num_classifiers,n", po::value<int>()->required(),
                     "Num classifiers");
  desc.add_options()("batch,b", po::value<int>()->required(), "Batch size");
  desc.add_options()("memory,a", po::value<bool>(), "memory");

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
  do_mem = args.count("memory");
  num_classifiers = args["num_classifiers"].as<int>();
  batch_size = args["batch"].as<int>();
  LOG(INFO) << batch_size;
  Run(camera_name, model);
  return 0;
}
