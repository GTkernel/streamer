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
std::vector<StreamReader *> mtcnn_output_readers;
std::vector<std::shared_ptr<GstVideoEncoder>> encoders;

void CleanUp() {
  for (auto encoder : encoders) {
    if (encoder->IsStarted()) encoder->Stop();
  }

  for (auto reader : mtcnn_output_readers) {
    reader->UnSubscribe();
  }

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

void Run(const std::vector<string> &camera_names, const string &model_name,
         bool display, float scale, int min_size) {
  cout << "Run face mtcnn demo" << endl;

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
  Shape input_shape(3, cameras[0]->GetWidth()*scale, cameras[0]->GetHeight()*scale);
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
  auto model_desc = model_manager.GetModelDescription(model_name);
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<Processor> mtcnn(new MtcnnFaceDetector(model_desc, min_size));
    mtcnn->SetSource("input", input_streams[i]);
    mtcnns.push_back(mtcnn);

    // mtcnn readers
    auto mtcnn_output = mtcnn->GetSink("output");
    mtcnn_output_readers.push_back(mtcnn_output->Subscribe());

    // encoders, encode each camera stream
    string output_filename = camera_names[i] + ".mp4";

    std::shared_ptr<GstVideoEncoder> encoder(new GstVideoEncoder(
        cameras[i]->GetWidth(), cameras[i]->GetHeight(), output_filename));
    encoder->SetSource("input", mtcnn->GetSink("output"));
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
      auto reader = mtcnn_output_readers[i];
      auto md_frame = reader->PopFrame<MetadataFrame>();
      if (display) {
        cv::Mat image = md_frame->GetOriginalImage();
        std::vector<FaceInfo> faceInfo = md_frame->GetFaceInfo();
        for(int i = 0;i<faceInfo.size();i++){
          float x = faceInfo[i].bbox.x1;
          float y = faceInfo[i].bbox.y1;
          float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
          float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
          cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),5);
        }
        for(int i=0;i<faceInfo.size();i++){
          FacePts facePts = faceInfo[i].facePts;
          for(int j=0;j<5;j++)
            cv::circle(image,cv::Point(facePts.y[j],facePts.x[j]),1,cv::Scalar(255,255,0),5);
        }
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
  auto model = vm["model"].as<string>();
  bool display = vm.count("display") != 0;
  float scale = vm["scale"].as<float>();
  int min_size = vm["min_size"].as<int>();
  Run(camera_names, model, display, scale, min_size);

  return 0;
}
