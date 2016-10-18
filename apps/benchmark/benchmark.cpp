/**
 * @brief benchmark.cpp - Used to run various benchmark of the system.
 */

#include <boost/program_options.hpp>
#include "tx1dnn.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

// Global arguments
bool verbose = false;
std::vector<string> processor_names;
std::vector<string> camera_names;
string experiment;
string net;
string framework;
int ITERATION;

void RunClassificationExperiment() {
  cout << "Run classification experiment" << endl;
  auto &model_manager = ModelManager::GetInstance();
  auto &camera_manager = CameraManager::GetInstance();

  int batch_size = camera_names.size();

  // Camera streams
  std::vector<std::shared_ptr<Camera>> cameras;
  for (auto camera_name : camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  std::vector<std::shared_ptr<Stream>> camera_streams;
  for (auto camera : cameras) {
    auto camera_stream = camera->GetStream();
    camera_streams.push_back(camera_stream);
  }

  // Input shape
  Shape input_shape(3, 227, 227);
  std::vector<std::shared_ptr<Stream>> input_streams;
  std::vector<std::shared_ptr<Processor>> processors;

  // transformers
  for (auto camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(new ImageTransformProcessor(
        camera_stream, input_shape, CROP_TYPE_CENTER,
        true /* subtract mean */));
    processors.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSinks()[0]);
  }

  // classifier
  auto model_desc = model_manager.GetModelDesc(net);
  std::shared_ptr<ImageClassificationProcessor> classifier(
      new ImageClassificationProcessor(input_streams, model_desc, input_shape));
  processors.push_back(classifier);

  for (auto camera : cameras) {
    camera->Start();
  }

  for (auto processor : processors) {
    processor->Start();
  }

  /////////////// RUN
  for (int itr = 1; itr <= ITERATION; itr++) {
    for (int i = 0; i < batch_size; i++) {
      auto stream = classifier->GetSinks()[i];
      auto md_frame = stream->PopMDFrame();
    }
    if (itr % 50 == 0) {
      if (verbose) {
        cout << "Run for " << itr << " iterations" << endl;
      }
    }
  }

  for (auto processor : processors) {
    processor->Stop();
  }

  for (auto camera : cameras) {
    camera->Stop();
  }
}

int main(int argc, char *argv[]) {
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  po::options_description desc("Benchmark for streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("net,n", po::value<string>()->value_name("NET"),
                     "The name of the neural net to run");
  desc.add_options()("framework,f",
                     po::value<string>()->value_name("FRAMEWORK"),
                     "The framework to run the neural net, either caffe, "
                     "caffefp16, gie, or mxnet");
  desc.add_options()("camera,c", po::value<string>()->value_name("CAMERAS"),
                     "The name of the camera to use, if there are multiple "
                     "cameras to be used, separate with ,");
  desc.add_options()("experiment,e", po::value<string>()->value_name("EXP"),
                     "Experriment to run");
  desc.add_options()("verbose,v", "Verbose logging or not");
  desc.add_options()("iter,i",
                     po::value<int>()->default_value(1000)->value_name("ITER"),
                     "Number of iterations to run");
  desc.add_options()(
      "pipeline,p", po::value<string>()->value_name("pipeline"),
      "The processor pipeline to run, separate processor with ,");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << e.what() << endl;
    cout << desc << endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("verbose")) {
    cout << "verbose" << endl;
    verbose = true;
  }

  //// Argument parsed

  if (vm.count("pipeline")) {
    auto pipeline = vm["pipeline"].as<string>();
    processor_names = SplitString(pipeline, ",");
  }

  if (vm.count("camera")) {
    auto camera = vm["camera"].as<string>();
    camera_names = SplitString(camera, ",");
  }

  if (vm.count("experiment")) {
    experiment = vm["experiment"].as<string>();
  }

  if (vm.count("net")) {
    net = vm["net"].as<string>();
  }

  if (vm.count("framework")) {
    framework = vm["framework"].as<string>();
  }

  ITERATION = vm["iter"].as<int>();

  if (experiment == "classification") {
    RunClassificationExperiment();
  }
}