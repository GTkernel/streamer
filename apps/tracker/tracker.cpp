#include <boost/program_options.hpp>
#include <csignal>
#include "streamer.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

/////// Global vars
std::vector<std::shared_ptr<Camera>> cameras;
std::vector<std::shared_ptr<Processor>> transformers;
std::vector<std::shared_ptr<Processor>> mtcnns;
std::shared_ptr<Facenet> facenet;
std::vector<std::shared_ptr<Processor>> trackers;
std::vector<StreamReader *> tracker_output_readers;
std::vector<std::shared_ptr<GstVideoEncoder>> encoders;

void CleanUp() {
  for (auto encoder : encoders) {
    if (encoder->IsStarted()) encoder->Stop();
  }

  for (auto reader : tracker_output_readers) {
    reader->UnSubscribe();
  }

  for (auto tracker : trackers) {
    if (tracker->IsStarted()) tracker->Stop();
  }

  if (facenet != nullptr && facenet->IsStarted()) facenet->Stop();

  for (auto mtcnn : mtcnns) {
    if (mtcnn->IsStarted()) mtcnn->Stop();
  }

  for (auto transformer : transformers) {
    if (transformer->IsStarted()) transformer->Stop();
  }

  for (auto camera : cameras) {
    if (camera->IsStarted()) camera->Stop();
  }
}

void SignalHandler(int signal) {
  std::cout << "Received SIGINT, try to gracefully exit" << std::endl;
  //  CleanUp();

  exit(0);
}

void Run(const std::vector<string> &camera_names, const string &mtcnn_model_name,
         const string &facenet_model_name, bool display, float scale, int min_size) {
  cout << "Run tracker demo" << endl;

  std::signal(SIGINT, SignalHandler);

  int batch_size = camera_names.size();
  CameraManager &camera_manager = CameraManager::GetInstance();
  ModelManager &model_manager = ModelManager::GetInstance();
  
// Check options
  CHECK(model_manager.HasModel(mtcnn_model_name)) << "Model " << mtcnn_model_name
                                            << " does not exist";
  CHECK(model_manager.HasModel(facenet_model_name)) << "Model " << facenet_model_name
                                            << " does not exist";
  for (auto camera_name : camera_names) {
    CHECK(camera_manager.HasCamera(camera_name)) << "Camera " << camera_name
                                                 << " does not exist";
  }

  ////// Start cameras, processors

  for (auto camera_name : camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  // Do video stream classification
  std::vector<std::shared_ptr<Stream>> camera_streams;
  for (auto camera : cameras) {
    auto camera_stream = camera->GetStream();
    camera_streams.push_back(camera_stream);
  }
  Shape input_shape(3, cameras[0]->GetWidth(), cameras[0]->GetHeight());
  std::vector<std::shared_ptr<Stream>> input_streams;

  // Transformers
  for (auto camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(new ImageTransformer(
        input_shape, CROP_TYPE_INVALID, false, false));
    transform_processor->SetSource("input", camera_stream);
    transformers.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSink("output"));
  }

  // mtcnn
  auto model_description = model_manager.GetModelDescription(mtcnn_model_name);
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<Processor> mtcnn(new MtcnnFaceDetector(model_description, min_size));
    mtcnn->SetSource("input", input_streams[i]);
    mtcnns.push_back(mtcnn);
  }

  // facenet
  auto model_desc = model_manager.GetModelDesc(facenet_model_name);
  Shape input_shape_facenet(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  facenet.reset(
      new Facenet(model_desc, input_shape_facenet, input_streams.size()));

  for (size_t i = 0; i < batch_size; i++) {
    facenet->SetInputStream(i, mtcnns[i]->GetSink("output"));
  }

  // tracker
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<Processor> tracker(new ObjectTracker());
    tracker->SetSource("input", facenet->GetSink("output" + std::to_string(i)));
    trackers.push_back(tracker);

    // tracker readers
    auto tracker_output = tracker->GetSink("output");
    tracker_output_readers.push_back(tracker_output->Subscribe());

    // encoders, encode each camera stream
    string output_filename = camera_names[i] + ".mp4";

    std::shared_ptr<GstVideoEncoder> encoder(new GstVideoEncoder(
        cameras[i]->GetWidth(), cameras[i]->GetHeight(), output_filename));
    encoder->SetSource("input", tracker->GetSink("output"));
    encoders.push_back(encoder);
  }

  for (auto camera : cameras) {
    if (!camera->IsStarted()) {
      camera->Start();
    }
  }

  for (auto transformer : transformers) {
    transformer->Start();
  }

  for (auto mtcnn : mtcnns) {
    mtcnn->Start();
  }

  facenet->Start();

  for (auto tracker : trackers) {
    tracker->Start();
  }

  for (auto encoder : encoders) {
    encoder->Start();
  }

  //////// Processor started, display the results

  if (display) {
    for (string camera_name : camera_names) {
      cv::namedWindow(camera_name, cv::WINDOW_NORMAL);
    }
  }

  //  double fps_to_show = 0.0;
  while (true) {
    for (int i = 0; i < camera_names.size(); i++) {
      auto reader = tracker_output_readers[i];
      auto image_frame = reader->PopFrame<ImageFrame>();
      if (display) {
        cv::Mat image = image_frame->GetImage();
        cv::imshow(camera_names[i], image);
      }
    }

    if (display) {
      int q = cv::waitKey(10);
      if (q == 'q') break;
    }
  }

  LOG(INFO) << "Done";

  //////// Clean up

  CleanUp();
  cv::destroyAllWindows();
}

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Multi-camera end to end video ingestion demo");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("mtcnn_model,m",
                     po::value<string>()->value_name("MTCNN_MODEL")->required(),
                     "The name of the mtcnn model to run");
  desc.add_options()("facenet_model",
                     po::value<string>()->value_name("FACENET_MODEL")->required(),
                     "The name of the facenet model to run");
  desc.add_options()("camera,c",
                     po::value<string>()->value_name("CAMERAS")->required(),
                     "The name of the camera to use, if there are multiple "
                     "cameras to be used, separate with ,");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()("device", po::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");
  desc.add_options()("scale,s", po::value<float>()->default_value(1.0),
                     "scale factor before mtcnn");
  desc.add_options()("min_size", po::value<int>()->default_value(40),
                     "face minimum size");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();
  int device_number = vm["device"].as<int>();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);

  auto camera_names = SplitString(vm["camera"].as<string>(), ",");
  auto mtcnn_model = vm["mtcnn_model"].as<string>();
  auto facenet_model = vm["facenet_model"].as<string>();
  bool display = vm.count("display") != 0;
  float scale = vm["scale"].as<float>();
  int min_size = vm["min_size"].as<int>();
  Run(camera_names, mtcnn_model, facenet_model, display, scale, min_size);

  return 0;
}
