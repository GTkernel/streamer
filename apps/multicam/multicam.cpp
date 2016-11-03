/**
 * @brief multicam.cpp - An example showing the usage to run realtime
 * classification on multiple camera streams. This example reads frames from
 * multiple cameras, overlays labels with each camera input, filter
 * `unimportant' videos and store the video and classification results locally.
 */

#include <boost/program_options.hpp>
#include <csignal>
#include "streamer.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

/////// Global vars
std::vector<std::shared_ptr<Camera>> cameras;
std::vector<std::shared_ptr<Processor>> transformers;
std::shared_ptr<ImageClassifier> classifier;
std::vector<StreamReader *> classifier_output_readers;
std::vector<std::shared_ptr<GstVideoEncoder>> encoders;

void CleanUp() {
  for (auto encoder : encoders) {
    if (encoder->IsStarted()) encoder->Stop();
  }

  for (auto reader : classifier_output_readers) {
    reader->UnSubscribe();
  }

  if (classifier != nullptr && classifier->IsStarted()) classifier->Stop();

  for (auto transformer : transformers) {
    if (transformer->IsStarted()) transformer->Stop();
  }

  for (auto camera : cameras) {
    if (camera->IsStarted()) camera->Stop();
  }
}

void SignalHandler(int signal) {
  std::cout << "Received SIGINT, try to gracefully exit" << std::endl;
  CleanUp();

  exit(0);
}

void Run(const std::vector<string> &camera_names, const string &model_name,
         bool display) {
  cout << "Run multicam demo" << endl;

  std::signal(SIGINT, SignalHandler);

  int batch_size = camera_names.size();
  CameraManager &camera_manager = CameraManager::GetInstance();
  ModelManager &model_manager = ModelManager::GetInstance();

  // Check options
  CHECK(model_manager.HasModel(model_name)) << "Model " << model_name
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

  Shape input_shape(3, 227, 227);
  std::vector<std::shared_ptr<Stream>> input_streams;

  // Transformers
  for (auto camera_stream : camera_streams) {
    std::shared_ptr<Processor> transform_processor(
        new ImageTransformer(camera_stream, input_shape, CROP_TYPE_CENTER,
                             true /* subtract mean */));
    transformers.push_back(transform_processor);
    input_streams.push_back(transform_processor->GetSinks()[0]);
  }

  // classifier
  auto model_desc = model_manager.GetModelDesc(model_name);
  classifier.reset(new ImageClassifier(input_streams, model_desc, input_shape));

  // classifier readers
  for (auto stream : classifier->GetSinks()) {
    classifier_output_readers.push_back(stream->Subscribe());
  }

  // encoders, encode each camera stream
  for (int i = 0; i < batch_size; i++) {
    string output_filename = camera_names[i] + ".mp4";

    std::shared_ptr<GstVideoEncoder> encoder(
        new GstVideoEncoder(classifier->GetSinks()[i], cameras[i]->GetWidth(),
                            cameras[i]->GetHeight(), output_filename));
    encoders.push_back(encoder);
  }

  for (auto camera : cameras) {
    camera->Start();
  }

  for (auto transformer : transformers) {
    transformer->Start();
  }

  classifier->Start();

  for (auto encoder : encoders) {
    encoder->Start();
  }

  //////// Processor started, display the results

  if (display) {
    for (string camera_name : camera_names) {
      cv::namedWindow(camera_name);
    }
  }

  int update_overlay = 0;
  const int UPDATE_OVERLAY_INTERVAL = 10;
  string label_to_show = "XXX";
  double fps_to_show = 0.0;
  while (true) {
    for (int i = 0; i < camera_names.size(); i++) {
      auto reader = classifier_output_readers[i];
      auto md_frame = reader->PopFrame<MetadataFrame>();
      if (display) {
        cv::Mat img = md_frame->GetOriginalImage();
        string label = md_frame->GetTags()[0];
        if (update_overlay == 1) {
          label_to_show = label;
          fps_to_show = classifier->GetAvgFps();
        }

        // Overlay FPS label and classification label
        double font_size = 0.8 * img.size[0] / 320.0;
        cv::Point label_point(img.rows / 6, img.cols / 3);
        cv::Scalar outline_color(0, 0, 0);
        cv::Scalar label_color(200, 200, 250);

        cv::putText(img, label_to_show, label_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, outline_color, 8, CV_AA);
        cv::putText(img, label_to_show, label_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, label_color, 2, CV_AA);

        cv::Point fps_point(img.rows / 3, img.cols / 6);

        char fps_string[256];
        sprintf(fps_string, "%.2lffps", fps_to_show);
        cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, outline_color, 8, CV_AA);
        cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                    font_size, label_color, 2, CV_AA);

        cv::imshow(camera_names[i], img);
      }
    }

    if (display) {
      int q = cv::waitKey(10);
      if (q == 'q') break;
    }

    update_overlay = (update_overlay + 1) % UPDATE_OVERLAY_INTERVAL;
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
  desc.add_options()("model,m",
                     po::value<string>()->value_name("MODEL")->required(),
                     "The name of the model to run");
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
  int device_number = vm["device"].as<int>();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_names = SplitString(vm["camera"].as<string>(), ",");
  auto model = vm["model"].as<string>();
  bool display = vm.count("display") != 0;
  Run(camera_names, model, display);

  return 0;
}
