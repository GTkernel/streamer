#include <sys/select.h>
#include <thread>

#include <boost/program_options.hpp>

// TODO: remove streamer.h
#include "streamer.h"
#include "processor/imagematch/imagematch.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"

namespace po = boost::program_options;

StreamPtr fake_nne;

// Function to send fake frames down a stream
// The reason we take a mat as a parameter is to avoid dealing with figuring out the
// correct size for the vishash
void FramePush(cv::Mat m) {
  unsigned long fid = 0;
  while(true) {
    usleep(500);
    auto frame = std::make_unique<Frame>();
    cv::randu(m, 0, 1);
    frame->SetValue("activations", m);
    frame->SetValue("frame_id", (unsigned long)fid++);
    frame->SetValue("Camera.Benchmark.StartTime", boost::posix_time::microsec_clock::local_time());
    frame->SetValue("NeuralNetEvaluator.Benchmark.Inference", (double)0);
    fake_nne->PushFrame(std::move(frame));
  }
}

void Run(const std::vector<string>& camera_names, const string& model_name,
         bool display, size_t batch_size, std::string linmod_path,
         std::string vishash_layer_name, bool do_linmod, std::string query_path,
         int num_query, int image_per_query, bool use_fake_nne=false) {
  std::stringstream csv_header;
  csv_header << "num queries" << ",";
  csv_header << "images per query" << ",";
  csv_header << "batch_size" << ",";
  csv_header << "NeuralNetInference" << ",";
  csv_header << "ImageMatchFull" << ",";
  csv_header << "ImageMatchMatrixMultiply" << ",";
  csv_header << "ImageMatchGatherAndAdd" << ",";
  csv_header << "ImageMatchLinearModelTrain" << ",";
  csv_header << "Overall Framerate" << ",";
  csv_header << "EndToEndLatency" << ",";

  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  // Check options
  CHECK(model_manager.HasModel(model_name))
      << "Model " << model_name << " does not exist";
  for (const auto& camera_name : camera_names) {
    CHECK(camera_manager.HasCamera(camera_name))
        << "Camera " << camera_name << " does not exist";
  }

  auto camera = camera_manager.GetCamera(camera_names.at(0));
  auto model_desc = model_manager.GetModelDesc(model_name);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  std::shared_ptr<Processor> transformer(new ImageTransformer(input_shape, true, true));
  auto entrance = std::make_shared<FlowControlEntrance>(10);
  std::vector<std::string> vishash_layer = {vishash_layer_name};
  auto neural_net_eval = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, batch_size, vishash_layer);
  auto imagematch = std::make_shared<ImageMatch>(linmod_path, do_linmod, batch_size);
  auto exit_proc = std::make_shared<FlowControlExit>();

  if(!use_fake_nne) {
    entrance->SetSource(camera->GetStream());
    transformer->SetSource("input", entrance->GetSink());
    neural_net_eval->SetSource("input", transformer->GetSink("output"), "");
    imagematch->SetSource("input", neural_net_eval->GetSink(vishash_layer_name));

    camera->Start();
    transformer->Start();
    neural_net_eval->Start();
  } else {
    fake_nne = std::make_shared<Stream>();
    entrance->SetSource(fake_nne);
    imagematch->SetSource("input", entrance->GetSink());
  }
  exit_proc->SetSource(imagematch->GetSink("output"));
  entrance->Start();
  exit_proc->Start();
  imagematch->Start();

  size_t vishash_size;
  cv::Mat vishash_mat;
  if(query_path != "") {
    cv::Mat img = cv::imread(query_path, 1);
    std::unique_ptr<Frame> input = std::make_unique<Frame>();
    input->SetValue("original_image", img);
    ImageTransformer* it = new ImageTransformer(input_shape, true, true);
    NeuralNetEvaluator* nne = new NeuralNetEvaluator(model_desc, input_shape, 1, vishash_layer);
    StreamPtr fake_input_stream = std::make_shared<Stream>();
    it->SetSource("input", fake_input_stream);
    nne->SetSource("input", it->GetSink("output"), "");
    it->Start();
    nne->Start();
    auto reader = nne->GetSink(vishash_layer_name)->Subscribe();
    fake_input_stream->PushFrame(std::move(input));
    auto frame = reader->PopFrame();
    vishash_mat = frame->GetValue<cv::Mat>("activations");
    std::vector<float> vishash(vishash_mat.begin<float>(), vishash_mat.end<float>());
    vishash_size = vishash.size();
    imagematch->SetQueryMatrix(num_query, image_per_query, vishash_size);
    /*for(int i = 0; i < num_query; ++i) {
      for(int j = 0; j < image_per_query; ++j) {
        imagematch->AddQuery(query_path, vishash, i, true);
      }
    }*/
  }
  if(use_fake_nne) {
    std::thread fake_nne_thread(FramePush, vishash_mat);
    fake_nne_thread.detach();
  }
  sleep(1);
  std::cout << csv_header.str() << std::endl;

  //////// Processor started, display the results

  if (display) {
    for (const auto& camera_name : camera_names) {
      cv::namedWindow(camera_name);
    }
  }

  int update_overlay = 0;
  const int UPDATE_OVERLAY_INTERVAL = 10;
  std::vector<string> label_to_show(camera_names.size());
  auto reader = exit_proc->GetSink()->Subscribe();
  while (true) {
    auto frame = reader->PopFrame();
    auto fps = reader->GetHistoricalFps();
    auto end_timestamp = boost::posix_time::microsec_clock::local_time();
    auto diff = end_timestamp - frame->GetValue<boost::posix_time::ptime>("Camera.Benchmark.StartTime");
    std::stringstream benchmark_summary;
    benchmark_summary << num_query << ",";
    benchmark_summary << image_per_query << ",";
    benchmark_summary << batch_size << ",";
    benchmark_summary << frame->GetValue<double>("NeuralNetEvaluator.Benchmark.Inference") << ",";
    benchmark_summary << frame->GetValue<double>("ImageMatch.Benchmark.EndToEnd") << ",";
    benchmark_summary << frame->GetValue<double>("ImageMatch.Benchmark.MatrixMultiply") << ",";
    benchmark_summary << frame->GetValue<double>("ImageMatch.Benchmark.GatherAndAdd") << ",";
    benchmark_summary << frame->GetValue<double>("ImageMatch.Benchmark.LinearModelTrain") << ",";
    benchmark_summary << fps << ",";
    benchmark_summary << diff.total_microseconds() << ",";
    std::cout << benchmark_summary.str() << std::endl;
    if (display) {
      cv::Mat img = frame->GetValue<cv::Mat>("image");

      // Overlay FPS label and classification label
      double font_size = 0.8 * img.size[0] / 320.0;
      cv::Point label_point(img.rows / 6, img.cols / 3);
      cv::Scalar outline_color(0, 0, 0);
      cv::Scalar label_color(200, 200, 250);

      cv::putText(img, frame->GetValue<std::string>("ImageMatchSummary"), label_point,
                  CV_FONT_HERSHEY_DUPLEX, font_size, label_color, 2, CV_AA);

      cv::Point fps_point(img.rows / 3, img.cols / 6);

      char fps_string[256];
      sprintf(fps_string, "%.2lffps", fps);
      cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                  font_size, outline_color, 8, CV_AA);
      cv::putText(img, fps_string, fps_point, CV_FONT_HERSHEY_DUPLEX,
                  font_size, label_color, 2, CV_AA);

      cv::imshow(camera_names.at(0), img);
    }

    struct timeval tv;
    fd_set fds;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    select(STDIN_FILENO+1, &fds, NULL, NULL, &tv);
    if(FD_ISSET(0, &fds)) {
      exit(0);
    }
    if (display) {
      int q = cv::waitKey(10);
      if (q == 'q') break;
    }

    update_overlay = (update_overlay + 1) % UPDATE_OVERLAY_INTERVAL;
  }

  LOG(INFO) << "Done";

  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
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
                     "The name of the camera to use");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()("use_fake_nne,u", "use fake NNE for max throughput or not");
  desc.add_options()("device", po::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()("batch_size,b", po::value<size_t>()->default_value(1),
                     "Batch size for inference");
  desc.add_options()("linear_model_path,l",
                     po::value<string>()->value_name("LINEAR_MODEL_PATH")->default_value(""),
                     "Path to linear model to use for imagematch");
  desc.add_options()("vishash_layer,v",
                     po::value<string>()->value_name("VISHASH_LAYER")->required(),
                     "Name of the Tensor to evaluate and use the output as the vishash");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");
  desc.add_options()("query_path,q",
                     po::value<std::string>()->value_name("QUERY_PATH")->default_value(""),
                     "Path to query image (single image, for testing purposes)");
  desc.add_options()("num_query,n",
                     po::value<int>()->value_name("NUM_QUERY")->default_value(1),
                     "Number of times to duplicate query");
  desc.add_options()("image_per_query,p",
                     po::value<int>()->value_name("IMAGE_PER_QUERY")->default_value(1),
                     "Number of times to duplicate image within query");
  desc.add_options()("fast_query_population,f",
                     "Use fast query population");
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

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }

  int device_number = vm["device"].as<int>();
  size_t batch_size = vm["batch_size"].as<size_t>();
  std::string linmod_path = vm["linear_model_path"].as<std::string>();
  std::string query_path = vm["query_path"].as<std::string>();
  std::string vishash_layer_name = vm["vishash_layer"].as<std::string>();
  auto camera_names = SplitString(vm["camera"].as<string>(), ",");
  auto model = vm["model"].as<string>();
  bool display = vm.count("display") != 0;
  int num_query = vm["num_query"].as<int>();
  int image_per_query = vm["image_per_query"].as<int>();
  bool use_fake_nne = vm.count("use_fake_nne") != 0;

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);

  Run(camera_names, model, display, batch_size, linmod_path, vishash_layer_name, linmod_path != "", query_path, num_query, image_per_query, use_fake_nne);
  return 0;
}
