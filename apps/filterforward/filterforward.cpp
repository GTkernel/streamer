
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/image_transformer.h"
#include "processor/imagematch/imagematch.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"
#include "processor/throttler.h"
#include "stream/frame.h"
#include "stream/stream.h"
#include "utils/string_utils.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

bool log_memory;
int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getPhysical(){ //Note: this value is in KB!
    if(log_memory == false)
      return 0;
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

int getVirtual(){ //Note: this value is in KB!
    if(log_memory == false)
      return 0;
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}


typedef struct {
  int num;
  std::string layer;
  int xmin;
  int ymin;
  int xmax;
  int ymax;
  bool flat;
  std::string path;
  float threshold;
} query_spec_t;

std::string kd_layer_name;
std::unordered_map<int, std::vector<query_spec_t>> the_map;

namespace po = boost::program_options;

// Used to signal all threads that the pipeline should stop.
std::atomic<bool> stopped(false);

// Designed to be run in its own thread. Sets "stopped" to true after num_frames
// have been processed or after a stop frame has been detected.
void Stopper(StreamPtr stream, unsigned int num_frames) {
  unsigned int count = 0;
  StreamReader* reader = stream->Subscribe();
  while (num_frames == 0 || ++count < num_frames + 1) {
    LOG(INFO) << count;
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr && frame->IsStopFrame()) {
      break;
    }
  }
  stopped = true;
  reader->UnSubscribe();
}

// Designed to be run in its own thread. Creates a log file containing
// performance metrics for the specified stream.
void Logger(size_t idx, StreamPtr stream, boost::posix_time::ptime log_time,
    std::vector<std::string> fields, const std::string& output_dir,
    const unsigned int num_frames, bool display = false) {
  cv::Mat current_image;
  cv::Mat last_match = cv::Mat::zeros(640, 480, CV_32F);
  if (display) {
    cv::namedWindow("current_image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("last_match", cv::WINDOW_AUTOSIZE);
  }
  std::ostringstream log;
  bool is_first_frame = true;
  double total_bytes = 0;
  boost::posix_time::ptime start_time;
  boost::posix_time::ptime previous_time;
  StreamReader* reader = stream->Subscribe();

  // Loop until the stopper thread signals that we need to stop.
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame == nullptr) {
      continue;
    } else if (frame->IsStopFrame()) {
      // We still need to check for stop frames, even though the stopper thread
      // is watching for stop frames at the highest level.
      break;
    } else {
      if (is_first_frame) {
        // We start recording after the first frame in order to reduce
        // experimental error when calculating the network bandwidth usage.
        is_first_frame = false;
        start_time = boost::posix_time::microsec_clock::local_time();
        previous_time = start_time;
      } else {
        current_image = frame->GetValue<cv::Mat>("original_image");
        total_bytes += frame->GetRawSizeBytes(
            std::unordered_set<std::string>{fields.begin(), fields.end()});

        // Calculate the network bandwidth.
        boost::posix_time::ptime current_time =
          boost::posix_time::microsec_clock::local_time();
        // Use total_microseconds() to avoid dividing by zero if less than a
        // second has passed.
        double net_bw_bps = (total_bytes * 8 /
            (current_time - start_time).total_microseconds()) *
          1000000;

        double fps = reader->GetHistoricalFps();

        long latency_micros =
          (current_time -
           frame->GetValue<boost::posix_time::ptime>("capture_time_micros"))
          .total_microseconds();

        // Assemble log message;
        std::ostringstream msg;
        if (frame->GetValue<std::vector<int>>("imagematch.matches").size() >=
            1) {
          if (display) {
            last_match = frame->GetValue<cv::Mat>("original_image");
          }
          net_bw_bps = 1;
        } else {
          net_bw_bps = 0;
        }

        long it_micros = frame->GetValue<long>("image_transformer.micros");
        long nne_micros = frame->GetValue<long>("neural_net_evaluator.inference_time_micros");
        long fug_micros = frame->GetValue<long>("fug.micros");
        long im_micros = frame->GetValue<long>("imagematch.end_to_end_time_micros");
        int physical_kb = getPhysical();
        int virtual_kb = getVirtual();
        msg << net_bw_bps << "," << fps << "," << latency_micros << "," << it_micros << "," << nne_micros << "," << fug_micros << "," << im_micros << "," << physical_kb << "," << virtual_kb;
        if (display) {
          cv::imshow("current_image", current_image);
          cv::imshow("last_match", last_match);
          cv::waitKey(1);
        }

        if ((current_time - previous_time).total_seconds() >= 2) {
          // Every two seconds, log a frame's metrics to the console so that the
          // user can verify that the program is making progress.
          previous_time = current_time;

          // State variables for progress bar
          const int progress_bar_len = 50;
          int string_pos = 0;

          std::cout << msg.str() << std::endl;
          log << msg.str() << std::endl;
        }
      }
    }
  }
  reader->UnSubscribe();

  std::ostringstream log_filepath;
  log_filepath << output_dir << "/ff_" << idx << "_"
    << boost::posix_time::to_iso_extended_string(log_time) << ".csv";
  std::ofstream log_file(log_filepath.str());
  log_file << "# network bandwidth (bps),fps,e2e latency (micros),Transformer micros,NNE micros,FV crop micros,imagematch micros,physical kb,virtual kb"
    << std::endl
    << log.str();
  log_file.close();
}

void Run(const std::string& ff_conf, unsigned int num_frames, bool block,
         size_t queue_size, const std::string& camera_name,
         unsigned int file_fps, int throttled_fps, unsigned int tokens,
         const std::string& model, std::vector<std::string> layers,
         size_t nne_batch_size, std::vector<std::string> fields,
         const std::string& output_dir, bool display) {
  boost::posix_time::ptime log_time =
      boost::posix_time::microsec_clock::local_time();

  boost::filesystem::path output_dir_path(output_dir);
  boost::filesystem::create_directory(output_dir_path);

  // Parse the ff_conf file.
  bool on_first_line = true;
  std::vector<std::pair<float, size_t>> buf_params;
  std::ifstream ff_conf_file(ff_conf);
  std::string line;
  unsigned int level_counter = 0;
  while (std::getline(ff_conf_file, line)) {
    std::vector<std::string> args = SplitString(line, ",");
    if (StartsWith(line, "#")) {
      // Ignore comment lines.
      continue;
    }
    CHECK(args.size() >= 3)
        << "Each line of the ff_conf file must contain at least three items: "
        << "Selectivity,BufferLength,NumQueries";

    float sel = std::stof(args.at(0));
    unsigned long buf_len = std::stoul(args.at(1));
    kd_layer_name = args.at(2);
    for (decltype(args.size()) i = 3; i < args.size();) {
      CHECK(i + 8 < args.size()) << "Malformed configuration file";
      the_map[level_counter].push_back(query_spec_t());
      query_spec_t* cur_query_spec = &(the_map[level_counter].back());
      cur_query_spec->num = atoi(args.at(i++).c_str());
      cur_query_spec->layer = args.at(i++);
      cur_query_spec->xmin = atoi(args.at(i++).c_str());
      cur_query_spec->ymin = atoi(args.at(i++).c_str());
      cur_query_spec->xmax = atoi(args.at(i++).c_str());
      cur_query_spec->ymax = atoi(args.at(i++).c_str());
      cur_query_spec->flat = args.at(i++) == "true";
      cur_query_spec->path = args.at(i++);
      cur_query_spec->threshold = std::stof(args.at(i++));
    }

    std::ostringstream msg;
    msg << "Level " << level_counter << " - ";
    if (on_first_line) {
      on_first_line = false;
    } else {
      buf_params.push_back({sel, (size_t)buf_len});
      msg << "selectivity: " << sel << ", buffer length: " << buf_len << ", ";
    }
    LOG(INFO) << msg.str();
    ++level_counter;
  }

  std::vector<std::shared_ptr<Processor>> procs;
  std::vector<std::thread> logger_threads;

  // Create a Camera.

  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  if (camera->GetCameraType() != CameraType::CAMERA_TYPE_GST) {
    throw(std::invalid_argument("Must use GST camera"));
  }
  std::shared_ptr<GSTCamera> gst_camera =
      std::dynamic_pointer_cast<GSTCamera>(camera);
  gst_camera->SetBlockOnPush(false);
  gst_camera->SetOutputFilepath(output_dir + "/" + camera_name + ".mp4");
  gst_camera->SetFileFramerate(file_fps);
  procs.push_back(gst_camera);
  StreamPtr camera_stream = gst_camera->GetStream();

  StreamPtr correct_fps_stream = camera_stream;
  if (throttled_fps > 0) {
    // If we are supposed to throttler the stream in software, then create a
    // Throttler.
    auto throttler = std::make_shared<Throttler>(throttled_fps);
    throttler->SetSource(camera_stream);
    throttler->SetBlockOnPush(block);
    procs.push_back(throttler);
    correct_fps_stream = throttler->GetSink();
  }

  // Create a FlowControlEntrance.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(tokens);
  fc_entrance->SetSource(correct_fps_stream);
  fc_entrance->SetBlockOnPush(block);
  procs.push_back(fc_entrance);

  // Create an ImageTransformer.
  ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource(fc_entrance->GetSink());
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create a NeuralNetEvaluator.
  std::vector<std::string> output_layer_names = layers;
  auto nne = std::make_shared<NeuralNetEvaluator>(
      model_desc, input_shape, nne_batch_size, output_layer_names);
  nne->SetSource(transformer->GetSink());
  nne->SetBlockOnPush(block);
  procs.push_back(nne);

  // auto fv_gen_kd = std::make_shared<FVGen>(

  auto fv_gen = std::make_shared<FvGen>();
  fv_gen->SetSource(nne->GetSink());
  fv_gen->SetBlockOnPush(block);
  procs.push_back(fv_gen);
  StreamPtr fvgen_stream = fv_gen->GetSink();

  // Create ImageMatch level 0. Use the same batch size as the
  // NeuralNetEvaluator.
  auto im_0 = std::make_shared<ImageMatch>(nne_batch_size);
  im_0->SetSource(fvgen_stream);
  im_0->SetBlockOnPush(block);

  auto& specs_vector = the_map[0];
  for (decltype(specs_vector.size()) i = 0; i < specs_vector.size(); ++i) {
    auto& cur_spec = specs_vector.at(i);
    for (int j = 0; j < cur_spec.num; ++j) {
      im_0->AddQuery(cur_spec.path, cur_spec.layer, cur_spec.threshold,
                     cur_spec.xmin, cur_spec.ymin, cur_spec.xmax, cur_spec.ymax,
                     cur_spec.flat);
      nne->PublishLayer(cur_spec.layer);
      fv_gen->AddFv(cur_spec.layer, cur_spec.xmin, cur_spec.ymin, cur_spec.xmax,
                    cur_spec.ymax, cur_spec.flat);
    }
  }
  procs.push_back(im_0);

  // Create a FlowControlExit.
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(im_0->GetSink());
  fc_exit->SetBlockOnPush(block);
  procs.push_back(fc_exit);

  // Create a logger thread to calculate statistics about the first ImageMatch
  // level's output stream.
  StreamPtr fc_exit_sink = fc_exit->GetSink();
  logger_threads.push_back(std::thread([fc_exit_sink, log_time, fields,
                                        output_dir, num_frames, display] {
    Logger(0, fc_exit_sink, log_time, fields, output_dir, num_frames, display);
  }));

  // Create additional keyframe detector + ImageMatch levels in the hierarchy.
  StreamPtr kd_input_stream = fvgen_stream;
  int count = 0;
  for (auto& param : buf_params) {
    LOG(INFO) << "Creating KeyframeDetector level " << count++
              << " with parameters: " << param.first << " " << param.second;
  }
  for (decltype(the_map.size()) i = 1; i < the_map.size(); ++i) {
    // Create a keyframe detector.
    std::pair<float, size_t> kd_buf_params = buf_params.at(i);
    std::vector<std::pair<float, size_t>> kd_buf_params_vec = {kd_buf_params};
    fv_gen->AddFv(kd_layer_name);
    nne->PublishLayer(kd_layer_name);
    auto kd =
        std::make_shared<KeyframeDetector>(kd_layer_name, kd_buf_params_vec);
    kd->SetSource(kd_input_stream);
    kd->SetBlockOnPush(block);
    procs.push_back(kd);
    kd_input_stream = kd->GetSink(kd->GetSinkName(0));

    // Create an ImageMatch.
    unsigned int kd_batch_size =
        ceil(kd_buf_params.first * kd_buf_params.second);
    auto additional_im = std::make_shared<ImageMatch>(kd_batch_size);
    additional_im->SetSource(kd->GetSink(kd->GetSinkName(0)));
    additional_im->SetBlockOnPush(block);
    // PUT CODE HERE
    auto& specs_vector = the_map[i];
    for (decltype(specs_vector.size()) j = 0; j < specs_vector.size(); ++j) {
      auto& cur_spec = specs_vector.at(j);
      for (int k = 0; k < cur_spec.num; ++k) {
        additional_im->AddQuery(
            cur_spec.path, cur_spec.layer, cur_spec.threshold, cur_spec.xmin,
            cur_spec.ymin, cur_spec.xmax, cur_spec.ymax, cur_spec.flat);
        nne->PublishLayer(cur_spec.layer);
        fv_gen->AddFv(cur_spec.layer, cur_spec.xmin, cur_spec.ymin,
                      cur_spec.xmax, cur_spec.ymax, cur_spec.flat);
      }
    }
    procs.push_back(additional_im);

    // Create a logger thread to calculate statistics about ImageMatch's output
    // stream.
    StreamPtr additional_im_sink = additional_im->GetSink();
    logger_threads.push_back(std::thread(
        [i, additional_im_sink, log_time, fields, output_dir, num_frames] {
          Logger(i + 1, additional_im_sink, log_time, fields, output_dir,
                 num_frames);
        }));
  }

  // Launch stopper thread.
  std::thread stopper_thread(
      [fc_exit_sink, num_frames] { Stopper(fc_exit_sink, num_frames); });

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start(queue_size);
  }

  while (!stopped) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Stop the processors in forward order.
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  exit(0);
  for (const auto& proc : procs) {
    proc->Stop();
  }

  // Join all of our helper threads.
  stopper_thread.join();
  for (auto& t : logger_threads) {
    t.join();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("FilterForward");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's configuration "
                     "files.");
  desc.add_options()("ff-conf,f", po::value<std::string>(),
                     "The file containing the keyframe detector's "
                     "configuration.");
  desc.add_options()("num-frames", po::value<unsigned int>()->default_value(0),
                     "The number of frames to run before automatically "
                     "stopping.");
  desc.add_options()("block,b",
                     "Whether processors should block when pushing frames.");
  desc.add_options()("queue-size,q", po::value<size_t>()->default_value(16),
                     "The size of the queues between processors.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("file-fps", po::value<unsigned int>()->default_value(30),
                     "Rate at which to read a file source (no effect if not "
                     "file source).");
  desc.add_options()("throttled-fps,i", po::value<int>()->default_value(0),
                     "The FPS at which to throttle (in software) the camera "
                     "stream. 0 means no throttling.");
  desc.add_options()("tokens,t", po::value<unsigned int>()->default_value(5),
                     "The number of flow control tokens to issue.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("layer,l", po::value<std::string>()->required(),
                     "The layer to extract and use as the basis for keyframe "
                     "detection and ImageMatch.");
  desc.add_options()("nne-batch-size,s", po::value<size_t>()->default_value(1),
                     "nne batch size");
  desc.add_options()("fields",
                     po::value<std::vector<std::string>>()
                         ->multitoken()
                         ->composing()
                         ->required(),
                     "The fields to send over the network");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to write output files.");
  desc.add_options()("memory-usage", "Log memory usage");
  desc.add_options()("display,d", "Enable display.");

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
  std::string ff_conf = args["ff-conf"].as<std::string>();
  unsigned int num_frames = args["num-frames"].as<unsigned int>();
  bool block = args.count("block");
  size_t queue_size = args["queue-size"].as<size_t>();
  std::string camera = args["camera"].as<std::string>();
  unsigned int file_fps = args["file-fps"].as<unsigned int>();
  int throttled_fps = args["throttled-fps"].as<int>();
  unsigned int tokens = args["tokens"].as<unsigned int>();
  std::string layer = args["layer"].as<std::string>();
  std::string model = args["model"].as<std::string>();
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  std::vector<std::string> fields =
      args["fields"].as<std::vector<std::string>>();
  std::string output_dir = args["output-dir"].as<std::string>();
  bool display = args.count("display");
  log_memory = args.count("memory-usage");
  Run(ff_conf, num_frames, block, queue_size, camera, file_fps, throttled_fps,
      tokens, model, {layer}, nne_batch_size, fields, output_dir, display);
  return 0;
}
