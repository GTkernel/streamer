/**
 * @brief classifier.cpp - An example application to classify images.
 */

#include <boost/program_options.hpp>
#include <csignal>

#include "streamer.h"

namespace po = boost::program_options;

std::shared_ptr<Camera> camera;
std::shared_ptr<ImageTransformer> transformer;
std::shared_ptr<ImageClassifier> classifier;

void SignalHandler(int) {
  std::cout << "Received SIGINT, stopping" << std::endl;
  if (classifier != nullptr) classifier->Stop();
  if (transformer != nullptr) transformer->Stop();
  if (camera != nullptr) camera->Stop();

  exit(0);
}

void Run(const string& camera_name, const string& net_name) {
  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  CHECK(camera_manager.HasCamera(camera_name))
      << "Camera " << camera_name << " does not exist";

  camera = camera_manager.GetCamera(camera_name);

  // Transformer
  Shape input_shape(3, 227, 227);
  transformer =
      std::make_shared<ImageTransformer>(input_shape, true /* subtract mean */);
  transformer->SetSource("input", camera->GetSink("bgr_output"));

  // Image classifier
  auto model_desc = model_manager.GetModelDesc(net_name);
  classifier = std::make_shared<ImageClassifier>(model_desc, input_shape, 1);
  classifier->SetSource("input", transformer->GetSink("output"));

  // Run
  camera->Start();
  transformer->Start();
  classifier->Start();

  LOG(INFO) << "Classifier running. CTRL-C to stop";

  while (true) {
  }
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Simple camera display test");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera",
                     po::value<string>()->value_name("CAMERA")->required(),
                     "The name of the camera to use");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR")->required(),
                     "The directory to find streamer's configurations");
  desc.add_options()("net,n",
                     po::value<string>()->value_name("NET")->required(),
                     "The name of the neural net to run");

  std::signal(SIGINT, SignalHandler);

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<string>();
  auto net_name = vm["net"].as<string>();

  // Run classifier
  Run(camera_name, net_name);

  return 0;
}
