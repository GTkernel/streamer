#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/image_classifier.h"
#include "processor/image_transformer.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, const std::string& net,
         const std::string& input_layer, const std::string& split_layer,
         const std::string& output_layer) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Camera
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Transformer
  auto model_desc = ModelManager::GetInstance().GetModelDesc(net);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer = std::make_shared<ImageTransformer>(input_shape, true);
  transformer->SetSource("input", camera->GetStream());
  procs.push_back(transformer);

  // NNE1
  std::vector<std::string> split_layers = {split_layer};
  auto nne1 = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                   split_layers);
  nne1->SetSource(transformer->GetSink(), input_layer);
  procs.push_back(nne1);

  // NNE2
  std::vector<std::string> output_layers = {output_layer};
  auto nne2 = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                   output_layers);
  nne2->SetSource(nne1->GetSink(), split_layer);
  procs.push_back(nne2);

  // add an classifier (NNC)
  auto classifier = std::make_shared<ImageClassifier>(model_desc, 1);
  classifier->SetSource("input", nne2->GetSink());
  procs.push_back(classifier);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  auto reader = classifier->GetSink("output")->Subscribe();
  std::cout << "Press \"Control-C\" to stop." << std::endl;

  while (true) {
    auto frame = reader->PopFrame();
    // Extract match percentage.
    auto probs = frame->GetValue<std::vector<double>>("probabilities");
    auto prob_percent = probs.front() * 100;

    // Extract tag.
    auto tags = frame->GetValue<std::vector<std::string>>("tags");
    auto tag = tags.front();
    std::regex re(".+? (.+)");
    std::smatch results;
    std::string tag_name;
    if (!std::regex_match(tag, results, re)) {
      tag_name = tag;
    } else {
      tag_name = results[1];
    }

    // Get Frame Rate
    double rate = reader->GetPushFps();

    std::ostringstream label;
    label.precision(2);
    label << rate << " FPS - " << prob_percent << "% - " << tag_name;
    auto label_string = label.str();
    std::cout << label_string << std::endl;
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Demonstrates splitting DNN evaluation across two NNEs");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("net,n", po::value<std::string>()->required(),
                     "The name of the neural net to run.");
  desc.add_options()("input,i", po::value<std::string>()->required(),
                     "The name of the input layer of the neural net.");
  desc.add_options()("split,s", po::value<std::string>()->required(),
                     "The name of the layer after which to split computation.");
  desc.add_options()("output,o", po::value<std::string>()->required(),
                     "The name of the output layer of the neural net.");

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
  std::string camera_name = args["camera"].as<std::string>();
  std::string net = args["net"].as<std::string>();
  std::string input = args["input"].as<std::string>();
  std::string split = args["split"].as<std::string>();
  std::string output = args["output"].as<std::string>();
  Run(camera_name, net, input, split, output);
  return 0;
}
