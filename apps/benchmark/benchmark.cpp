/**
 * @brief benchmark.cpp - Used to run various benchmark of the system.
 */

#include "streamer.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using std::cout;
using std::endl;

// Global arguments
struct Configurations {
  // Enable verbose logging or not
  bool verbose;
  // The name of processors to use
  std::vector<string> processor_names;
  // The name of cameras to use
  std::vector<string> camera_names;
  // The encoder to use
  string encoder;
  // The decoder to use
  string decoder;
  // The name of the experiment to run
  string experiment;
  // The network model to use
  string net;
  // Duration of a test
  int time;
  // Device number
  int device_number;
  // Store the video or not
  bool store;
  // Try to use fp16 or not
  bool usefp16;
  // Enable batch or not
  bool batch;
  // Batch size for NN inference experiment
  int batch_size;
} CONFIG;

void SLEEP(int sleep_time_in_s) {
  while (sleep_time_in_s >= 10) {
    cout << sleep_time_in_s << " to sleep" << endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    sleep_time_in_s -= 10;
  }
  std::this_thread::sleep_for(std::chrono::seconds(sleep_time_in_s));
}

/**
 * @brief Run end-to-end camera(s)->classifier(NN)->store pipeline.
 */
void RunEndToEndExperiment() {
  cout << "Run End To End Experiment" << endl;
  // Check argument
  CHECK(CONFIG.camera_names.size() != 0) << "You must give at least one camera";
  CHECK(CONFIG.net != "") << "You must specify the network by -n or --network";

  auto &model_manager = ModelManager::GetInstance();
  auto &camera_manager = CameraManager::GetInstance();

  auto camera_size = CONFIG.camera_names.size();

  // Camera streams
  std::vector<std::shared_ptr<Camera>> cameras;
  for (const auto &camera_name : CONFIG.camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  std::vector<std::shared_ptr<Stream>> camera_streams;
  for (const auto &camera : cameras) {
    auto camera_stream = camera->GetStream();
    camera_streams.push_back(camera_stream);
  }

  // Input shape
  Shape input_shape(3, 227, 227);
  std::vector<std::shared_ptr<Stream>> input_streams;
  std::vector<std::shared_ptr<Processor>> transformers;
  std::vector<std::shared_ptr<GstVideoEncoder>> encoders;

  // transformers
  for (const auto &camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(
        new ImageTransformer(input_shape, true /* subtract mean */));
    transform_processor->SetSource("input", camera_stream);
    transformers.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSink("output"));
  }

  // classifier
  std::vector<ProcessorPtr> classifiers;
  if (CONFIG.batch) {
    auto model_desc = model_manager.GetModelDesc(CONFIG.net);
    std::shared_ptr<ImageClassifier> classifier(
        new ImageClassifier(model_desc, input_shape, input_streams.size()));

    for (decltype(input_streams.size()) i = 0; i < input_streams.size(); ++i) {
      classifier->SetInputStream(i, input_streams[i]);
    }
    classifiers.push_back(classifier);
  } else {
    auto model_desc = model_manager.GetModelDesc(CONFIG.net);
    for (const auto &s : input_streams) {
      std::shared_ptr<ImageClassifier> classifier(
          new ImageClassifier(model_desc, input_shape, 1));
      classifier->SetInputStream(0, s);
      classifiers.push_back(classifier);
    }
  }

  // encoders, encode each camera stream
  if (CONFIG.store) {
    if (CONFIG.batch) {
      auto classifier = classifiers[0];
      for (decltype(camera_size) i = 0; i < camera_size; ++i) {
        string output_filename = CONFIG.camera_names[i] + ".mp4";

        std::shared_ptr<GstVideoEncoder> encoder(new GstVideoEncoder(
            cameras[i]->GetWidth(), cameras[i]->GetHeight(), output_filename));
        encoder->SetSource("input",
                           classifier->GetSink("output" + std::to_string(i)));
        encoders.push_back(encoder);
      }
    } else {
      for (decltype(camera_size) i = 0; i < camera_size; ++i) {
        auto classifier = classifiers[i];
        string output_filename = CONFIG.camera_names[i] + ".mp4";

        std::shared_ptr<GstVideoEncoder> encoder(new GstVideoEncoder(
            cameras[i]->GetWidth(), cameras[i]->GetHeight(), output_filename));
        encoder->SetSource("input",
                           classifier->GetSink("output" + std::to_string(0)));
        encoders.push_back(encoder);
      }
    }
  }

  for (const auto &camera : cameras) {
    camera->Start();
  }

  for (const auto &transformer : transformers) {
    transformer->Start();
  }

  for (const auto &classifier : classifiers) {
    classifier->Start();
  }

  for (const auto &encoder : encoders) {
    encoder->Start();
  }

  /////////////// RUN
  SLEEP(CONFIG.time);

  for (const auto &encoder : encoders) {
    encoder->Stop();
  }

  for (const auto &classifier : classifiers) {
    classifier->Stop();
  }

  for (const auto &transformer : transformers) {
    transformer->Stop();
  }

  for (const auto &camera : cameras) {
    camera->Stop();
  }

  /////////////// PRINT STATS
  for (decltype(cameras.size()) i = 0; i < cameras.size(); ++i) {
    cout << "-- camera[" << i << "] fps is " << cameras[i]->GetAvgFps() << endl;
  }
  for (decltype(transformers.size()) i = 0; i < transformers.size(); ++i) {
    cout << "-- transformer[" << i << "] fps is "
         << transformers[i]->GetAvgFps() << endl;
  }

  for (decltype(classifiers.size()) i = 0; i < classifiers.size(); ++i) {
    cout << "-- classifier << " << i << " fps is "
         << classifiers[i]->GetAvgFps() << endl;
  }

  if (CONFIG.store) {
    for (decltype(encoders.size()) i = 0; i < encoders.size(); ++i) {
      cout << "-- encoder[" << i << "] fps is " << encoders[i]->GetAvgFps()
           << endl;
    }
  }
}

/**
 * @brief Benchmark the time to take the forward pass or a neural network
 */
void RunNNInferenceExperiment() {
  cout << "Run NN Inference Experiment" << endl;
  // Check argument
  CHECK(CONFIG.net != "") << "You must specify the network by -n or --network";

  auto &model_manager = ModelManager::GetInstance();

  auto model_desc = model_manager.GetModelDesc(CONFIG.net);
  std::shared_ptr<DummyNNProcessor> dummy_processor(
      new DummyNNProcessor(model_desc, CONFIG.batch_size));

  dummy_processor->Start();

  SLEEP(CONFIG.time);

  dummy_processor->Stop();
  cout << "-- processor fps is " << dummy_processor->GetAvgFps() << endl;
}

int main(int argc, char *argv[]) {
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Benchmark for streamer");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("net,n", po::value<string>()->value_name("NET"),
                     "The name of the neural net to run");
  desc.add_options()("camera,c", po::value<string>()->value_name("CAMERAS"),
                     "The name of the camera to use, if there are multiple "
                     "cameras to be used, separate with ,");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configuration");
  desc.add_options()("experiment,e",
                     po::value<string>()->value_name("EXP")->required(),
                     "Experiment to run");
  desc.add_options()("verbose,v", po::value<bool>()->default_value(false),
                     "Verbose logging or not");
  desc.add_options()("encoder", po::value<string>(), "Encoder to use");
  desc.add_options()("decoder", po::value<string>(), "Decoder to use");
  desc.add_options()("time,t", po::value<int>()->default_value(10),
                     "Duration of the experiment");
  desc.add_options()("device", po::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()("fp16", po::value<bool>()->default_value(false),
                     "Try to use fp16 when possible");
  desc.add_options()(
      "pipeline,p", po::value<string>()->value_name("pipeline"),
      "The processor pipeline to run, separate processor with ,");
  desc.add_options()(
      "batch_size",
      po::value<int>()->value_name("BATCH_SIZE")->default_value(1),
      "The batch used to benchmark a network");
  desc.add_options()(
      "batch",
      po::value<bool>()->value_name("ENABLE_BATCH")->default_value(false),
      "Enable batch or not in end to end experiment");
  desc.add_options()("store", po::value<bool>()->default_value(false),
                     "Write video at the end of the pipeline or not");

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
    CONFIG.verbose = true;
  }

  //// Prase arguments

  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  if (vm.count("pipeline")) {
    auto pipeline = vm["pipeline"].as<string>();
    CONFIG.processor_names = SplitString(pipeline, ",");
  }

  if (vm.count("camera")) {
    auto camera = vm["camera"].as<string>();
    CONFIG.camera_names = SplitString(camera, ",");
  }

  if (vm.count("experiment")) {
    CONFIG.experiment = vm["experiment"].as<string>();
  }

  if (vm.count("net")) {
    CONFIG.net = vm["net"].as<string>();
  }

  if (vm.count("encoder")) {
    CONFIG.encoder = vm["encoder"].as<string>();
    Context::GetContext().SetString(H264_ENCODER_GST_ELEMENT, CONFIG.encoder);
  }

  if (vm.count("decoder")) {
    CONFIG.decoder = vm["decoder"].as<string>();
    Context::GetContext().SetString(H264_DECODER_GST_ELEMENT, CONFIG.decoder);
  }

  if (vm.count("store")) {
    CONFIG.store = vm["store"].as<bool>();
  }

  CONFIG.verbose = vm["verbose"].as<bool>();
  CONFIG.time = vm["time"].as<int>();
  CONFIG.device_number = vm["device"].as<int>();
  CONFIG.batch_size = vm["batch_size"].as<int>();
  CONFIG.usefp16 = vm["fp16"].as<bool>();
  CONFIG.batch = vm["batch"].as<bool>();
  Context::GetContext().SetInt(DEVICE_NUMBER, CONFIG.device_number);
  Context::GetContext().SetBool(USEFP16, CONFIG.usefp16);

  if (CONFIG.experiment == "endtoend") {
    RunEndToEndExperiment();
  } else if (CONFIG.experiment == "nninfer") {
    RunNNInferenceExperiment();
  } else {
    LOG(ERROR) << "Unkown experiment: " << CONFIG.experiment;
  }
}
