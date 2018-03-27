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

// Server to run and update pipelines

#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <json/src/json.hpp>
#include <zmq.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/image_transformer.h"
#include "processor/neural_net_evaluator.h"
#include "processor/throttler.h"

namespace po = boost::program_options;

std::shared_ptr<Camera> camera;
std::shared_ptr<ImageTransformer> transformer;

void SignalHandler(int) {
  std::cout << "Received SIGINT, stopping" << std::endl;
  if (transformer != nullptr) transformer->Stop();
  if (camera != nullptr) camera->Stop();

  exit(0);
}

void Run(const std::string& camera_name, const std::string& net_name,
         const std::string& models_dir) {
  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  camera = camera_manager.GetCamera(camera_name);
  camera->SetBlockOnPush(true);

  // Start ZMQ server
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);
  socket.bind("tcp://*:5555");

  // Transformer
  Shape input_shape(3, 299, 299);
  transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetSink("output"));
  transformer->SetBlockOnPush(true);

  std::vector<std::shared_ptr<NeuralNetEvaluator>> nnes;
  std::vector<std::shared_ptr<NeuralNetEvaluator>> task_nnes;
  std::vector<std::shared_ptr<Throttler>> throttlers;

  // Run
  camera->Start();
  transformer->Start();

  while (true) {
    // Check for zmq message
    zmq::message_t request;
    int rc = socket.recv(&request, ZMQ_NOBLOCK);
    if (rc == 0 && zmq_errno() == EAGAIN) {
      // No ZMQ message
      continue;
    }

    //  Handle message
    std::cout << "ZMQ client message received" << std::endl;
    std::string msg =
        std::string(static_cast<char*>(request.data()), request.size());
    auto msg_json = nlohmann::json::parse(msg);

    // Stop old NNEs
    for (const auto& nne : nnes) {
      if (nne->IsStarted()) nne->Stop();
    }
    for (const auto& throttler : throttlers) {
      if (throttler->IsStarted()) throttler->Stop();
    }
    nnes.clear();
    task_nnes.clear();
    throttlers.clear();

    // Start new NNEs
    std::unordered_map<int, std::shared_ptr<NeuralNetEvaluator>> sink_by_id;
    for (const auto& nne_desc : msg_json) {
      int nne_id = nne_desc["net_id"];

      std::string input_layer = nne_desc["input_layer"];
      std::string output_layer = nne_desc["output_layer"];
      std::vector<std::string> output_layers;
      output_layers.push_back(output_layer);
      int parent_processor_id = nne_desc["parent_id"];
      StreamPtr sink;
      if (parent_processor_id == -1) {
        sink = transformer->GetSink("output");
      } else {
        auto it = sink_by_id.find(parent_processor_id);
        if (it == sink_by_id.end()) {
          LOG(ERROR) << "Could not find sink for nne " << parent_processor_id;
        }
        sink = it->second->GetSink(input_layer);
      }

      Shape input_shape(nne_desc["channels"], nne_desc["width"],
                        nne_desc["height"]);

      // Update model path
      std::string model_path = nne_desc["model_path"];
      std::string full_model_path = models_dir + "/" + model_path;
      auto base_model_desc = model_manager.GetModelDesc(net_name);
      auto new_model_desc = ModelDesc(
          base_model_desc.GetName(), base_model_desc.GetModelType(),
          full_model_path, base_model_desc.GetModelParamsPath(),
          base_model_desc.GetInputWidth(), base_model_desc.GetInputHeight(),
          base_model_desc.GetDefaultInputLayer(),
          base_model_desc.GetDefaultOutputLayer());

      // Make throttler and set source
      int target_fps = nne_desc["target_fps"];
      std::shared_ptr<Throttler> throttler =
          std::make_shared<Throttler>(target_fps);
      throttler->SetSource("input", sink);

      // Make nne and set source
      // TODO batch size
      std::shared_ptr<NeuralNetEvaluator> nn =
          std::make_shared<NeuralNetEvaluator>(new_model_desc, input_shape, 1,
                                               output_layers);
      nn->SetSource("input", throttler->GetSink("output"), input_layer);
      nn->SetBlockOnPush(true);

      // Add to sinks by name and nnes list
      nnes.push_back(nn);
      sink_by_id.insert({nne_id, nn});
      throttlers.push_back(throttler);

      bool shared = nne_desc["shared"].get<bool>();
      if (!shared) {
        task_nnes.push_back(nn);
      }
    }

    sink_by_id.clear();

    for (const auto& throttler : throttlers) throttler->Start();
    for (const auto& nne : nnes) nne->Start();

    LOG(INFO) << "Pipeline started.\n";

    // Log metric in 30 seconds
    double last_ms = Context::GetContext().GetTimer().ElapsedMSec();
    double now = Context::GetContext().GetTimer().ElapsedMSec();
    while (now - last_ms < 30000) {
      now = Context::GetContext().GetTimer().ElapsedMSec();
    }
    std::string message;
    std::string comma = "";
    for (const auto& throttler : throttlers) {
      double throughput = throttler->GetHistoricalProcessFps();
      std::cout << "Throttler: " << throughput << std::endl;
    }
    for (const auto& nne : nnes) {
      std::cout << "Processor: " << nne->GetHistoricalProcessFps() << std::endl;
    }
    for (const auto& task_nne : task_nnes) {
      double throughput = task_nne->GetHistoricalProcessFps();
      message += comma + std::to_string(throughput);
      comma = ",";
    }
    std::cout << message << std::endl;

    // Respond to client
    zmq::message_t reply(message.size());
    memcpy(reply.data(), message.c_str(), message.size());
    socket.send(reply);
  }

  return;
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple camera display test");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera",
                     po::value<std::string>()->value_name("CAMERA")->required(),
                     "The name of the camera to use");
  desc.add_options()(
      "config_dir,C",
      po::value<std::string>()->value_name("CONFIG_DIR")->required(),
      "The directory to find streamer's configurations");
  desc.add_options()("net,n",
                     po::value<std::string>()->value_name("NET")->required(),
                     "The name of the neural net to run");
  desc.add_options()("models,m",
                     po::value<std::string>()->value_name("MODELS")->required(),
                     "The name of the neural net to run");

  std::signal(SIGINT, SignalHandler);

  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, desc), args);
    po::notify(args);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (args.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Parse arguments
  if (args.count("config_dir")) {
    Context::GetContext().SetConfigDir(args["config_dir"].as<std::string>());
  }

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = args["camera"].as<std::string>();
  auto net_name = args["net"].as<std::string>();
  auto models_dir = args["models"].as<std::string>();

  // Run mainstream server
  Run(camera_name, net_name, models_dir);

  return 0;
}
