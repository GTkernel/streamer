/**
 * @brief mainstream_performance: Application used to benchmark Mainstream
 */

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/image_transformer.h"
#include "processor/neural_net_evaluator.h"

namespace po = boost::program_options;

// Global arguments
struct Configurations {
  std::vector<std::string> camera_names;
  std::string net;
  int num_apps;
  std::string outfile;
  bool resnet;
  bool inception;
  bool mobilenets;
} CONFIG;

static const std::string arr_mnets[] = {"input_1",
                                        "conv1_relu/clip_by_value",
                                        "conv_pw_1_relu/clip_by_value",
                                        "conv_pw_2_relu/clip_by_value",
                                        "conv_pw_3_relu/clip_by_value",
                                        "conv_pw_4_relu/clip_by_value",
                                        "conv_pw_5_relu/clip_by_value",
                                        "conv_pw_6_relu/clip_by_value",
                                        "conv_pw_7_relu/clip_by_value",
                                        "conv_pw_8_relu/clip_by_value",
                                        "conv_pw_9_relu/clip_by_value",
                                        "conv_pw_10_relu/clip_by_value",
                                        "conv_pw_11_relu/clip_by_value",
                                        "conv_pw_12_relu/clip_by_value",
                                        "conv_pw_13_relu/clip_by_value",
                                        "conv_pw_13_relu/clip_by_value",
                                        "dense_2/Softmax:0"};

static const std::string arr_iv3[] = {"input_1",
                                      "conv2d_1/convolution",
                                      "conv2d_2/convolution",
                                      "conv2d_3/convolution",
                                      "max_pooling2d_1/MaxPool",
                                      "conv2d_4/convolution",
                                      "conv2d_5/convolution",
                                      "max_pooling2d_2/MaxPool",
                                      "mixed0/concat",
                                      "mixed1/concat",
                                      "mixed2/concat",
                                      "mixed3/concat",
                                      "mixed4/concat",
                                      "mixed5/concat",
                                      "mixed6/concat",
                                      "mixed7/concat",
                                      "mixed8/concat",
                                      "mixed9/concat",
                                      "mixed10/concat",
                                      "dense_2/Softmax:0"};

static const std::string arr_r50[] = {"input_1",
                                      "conv1/BiasAdd",
                                      "bn_conv1/batchnorm/add_1",
                                      "activation_1/Relu",
                                      "max_pooling2d_1/MaxPool",
                                      "activation_4/Relu",
                                      "activation_7/Relu",
                                      "activation_10/Relu",
                                      "activation_13/Relu",
                                      "activation_16/Relu",
                                      "activation_19/Relu",
                                      "activation_22/Relu",
                                      "activation_25/Relu",
                                      "activation_28/Relu",
                                      "activation_31/Relu",
                                      "activation_34/Relu",
                                      "activation_37/Relu",
                                      "activation_40/Relu",
                                      "activation_43/Relu",
                                      "activation_46/Relu",
                                      "activation_49/Relu",
                                      "dense_2/Softmax:0"};

std::vector<std::string> layers_iv3(arr_iv3,
                                    arr_iv3 +
                                        sizeof(arr_iv3) / sizeof(arr_iv3[0]));
std::vector<std::string> layers_r50(arr_r50,
                                    arr_r50 +
                                        sizeof(arr_r50) / sizeof(arr_r50[0]));
std::vector<std::string> layers_mnets(arr_mnets, arr_mnets +
                                                     sizeof(arr_mnets) /
                                                         sizeof(arr_mnets[0]));

void MeasurePerformance(const std::string split_layer, int num_apps,
                        std::string file_prefix) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Make camera
  auto& camera_manager = CameraManager::GetInstance();
  auto camera_name = CONFIG.camera_names[0];
  auto camera = camera_manager.GetCamera(camera_name);
  procs.push_back(camera);

  // FlowControlEntrance
  auto entrance = std::make_shared<FlowControlEntrance>(4);
  entrance->SetSource(camera->GetStream());
  procs.push_back(entrance);

  // Make model
  auto& model_manager = ModelManager::GetInstance();
  auto model_desc = model_manager.GetModelDesc(CONFIG.net);

  Shape input_shape(3, 299, 299);

  // Make transformer
  std::shared_ptr<ImageTransformer> transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource("input", entrance->GetSink());
  procs.push_back(transformer);

  // Make NNEs
  std::string input1 = "input_1";
  std::string output1 = split_layer;
  std::string input2 = split_layer;
  std::string output2 = "dense_2/Softmax:0";

  std::vector<std::string> output_layers1;
  output_layers1.push_back(output1);
  std::vector<std::string> output_layers2;
  output_layers2.push_back(output2);

  std::shared_ptr<NeuralNetEvaluator> base_nne = NULL;
  std::vector<std::shared_ptr<NeuralNetEvaluator>> task_nnes;

  // TODO: Make it possible for NN to be complete pass through
  std::vector<std::shared_ptr<NeuralNetEvaluator>> endpoint_nnes;
  if (split_layer == "input_1") {
    // All NNs should be application specific
    for (int i = 0; i < num_apps; i++) {
      std::shared_ptr<NeuralNetEvaluator> nne =
          std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                               output_layers2);
      nne->SetSource("input", transformer->GetSink("output"), input1);
      procs.push_back(nne);

      task_nnes.push_back(nne);
      endpoint_nnes.push_back(nne);
    }
  } else if (split_layer == "dense_2/Softmax:0") {
    // Only base NN is needed
    base_nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                    output_layers1);
    base_nne->SetSource("input", transformer->GetSink("output"), input1);
    procs.push_back(base_nne);
    endpoint_nnes.push_back(base_nne);
  } else {
    // Both task and base NNs are needed
    base_nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                    output_layers1);
    base_nne->SetSource("input", transformer->GetSink("output"), input1);
    procs.push_back(base_nne);

    for (int i = 0; i < num_apps; i++) {
      std::shared_ptr<NeuralNetEvaluator> nne =
          std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                               output_layers2);
      nne->SetSource("input", base_nne->GetSink(output1), input2);
      procs.push_back(nne);

      task_nnes.push_back(nne);

      endpoint_nnes.push_back(nne);
    }
  }

  // FlowControlExit
  auto exit = std::make_shared<FlowControlExit>();
  for (auto const& endpoint : endpoint_nnes) {
    exit->SetSource(endpoint->GetSink(output2));
  }
  procs.push_back(exit);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  sleep(30);

  std::ofstream output_latency;
  std::ofstream output_fps;
  output_latency.open(file_prefix + "-latency", std::ios_base::app);
  output_fps.open(file_prefix + "-fps", std::ios_base::app);

  output_latency << split_layer << "," << num_apps;
  output_fps << split_layer << "," << num_apps;
  if (base_nne) {
    output_latency << "," << base_nne->GetAvgProcessingLatencyMs();
    output_fps << "," << base_nne->GetHistoricalProcessFps();
  } else {
    output_latency << ",0";
    output_fps << ",0";
  }
  for (const auto& task_nne : task_nnes) {
    output_latency << "," << task_nne->GetAvgProcessingLatencyMs();
    output_fps << "," << task_nne->GetHistoricalProcessFps();
  }

  output_latency << "\n";
  output_fps << "\n";
  output_latency.close();
  output_fps.close();

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }

  return;
}

void RunPerformanceExperiment() {
  CHECK(CONFIG.camera_names.size() == 1) << "You must give one camera";
  CHECK(CONFIG.net != "") << "You must specify the network by -n or --network";
  CHECK(CONFIG.num_apps != -1)
      << "You must specify the number of classifiers by --numapps";
  CHECK(CONFIG.outfile != "") << "You must specify the outfile for results";

  std::vector<std::string> layers;
  if (CONFIG.resnet)
    layers = layers_r50;
  else if (CONFIG.inception)
    layers = layers_iv3;
  else if (CONFIG.mobilenets)
    layers = layers_mnets;
  else
    LOG(ERROR) << "You must specify resnet, inception or mobilenets";

  for (int i = 1; i <= CONFIG.num_apps; i++) {
    for (const auto& split_layer : layers)
      MeasurePerformance(split_layer, i, CONFIG.outfile);
  }
}

int main(int argc, char* argv[]) {
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Benchmark for streamer");
  desc.add_options()("net,n", po::value<std::string>()->value_name("NET"),
                     "The name of the neural net to run");
  desc.add_options()("camera,c",
                     po::value<std::string>()->value_name("CAMERAS"),
                     "Name of camera to use");
  desc.add_options()("config_dir,C",
                     po::value<std::string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configuration");
  desc.add_options()("numapps", po::value<int>()->default_value(-1),
                     "Number classifiers to run");
  desc.add_options()("outfile", po::value<std::string>()->default_value(""),
                     "File to store results");
  desc.add_options()("resnet,r", "Use ResNet50 architecture or not");
  desc.add_options()("inception,i", "Use InceptionV3 architecture or not");
  desc.add_options()("mobilenets,m", "Use MobileNets architecture or not");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  //// Parse arguments

  if (vm.count("config_dir"))
    Context::GetContext().SetConfigDir(vm["config_dir"].as<std::string>());
  Context::GetContext().Init();

  if (vm.count("camera")) {
    auto camera = vm["camera"].as<std::string>();
    CONFIG.camera_names = SplitString(camera, ",");
  }

  if (vm.count("net")) CONFIG.net = vm["net"].as<std::string>();
  if (vm.count("numapps")) CONFIG.num_apps = vm["numapps"].as<int>();
  if (vm.count("outfile")) CONFIG.outfile = vm["outfile"].as<std::string>();
  if (vm.count("resnet")) CONFIG.resnet = vm.count("resnet") != 0;
  if (vm.count("inception")) CONFIG.inception = vm.count("inception") != 0;
  if (vm.count("mobilenets")) CONFIG.mobilenets = vm.count("mobilenets") != 0;

  RunPerformanceExperiment();
}
