
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
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
#include "processor/frame_writer.h"
#include "processor/image_transformer.h"
#include "processor/imagematch/imagematch.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"
#include "processor/throttler.h"
#include "stream/frame.h"
#include "stream/stream.h"
#include "utils/string_utils.h"

namespace po = boost::program_options;

// Used to signal all threads that the pipeline should stop.
std::atomic<bool> stopped(false);

// Designed to be run in its own thread. Detects when a stop frame is sent to
// the provided stream and sets "stopped" to true.
void Stopper(StreamPtr highest_stream) {
  auto reader = highest_stream->Subscribe();
  while (true) {
    if (reader->PopFrame()->IsStopFrame()) {
      stopped = true;
      break;
    }
  }
  reader->UnSubscribe();
}

void Logger(size_t idx, StreamPtr stream, boost::posix_time::ptime log_time,
            const std::string& output_dir) {
  // Loop until the stopper thread signals that we need to stop.
  std::ostringstream log_msg;
  bool is_first_sample = true;
  double total_bytes = 0;
  boost::posix_time::ptime start_time;
  boost::posix_time::ptime previous_time;
  auto reader = stream->Subscribe();

  while (!stopped) {
    auto frame = reader->PopFrame();
    if (!frame->IsStopFrame()) {
      if (is_first_sample) {
        is_first_sample = false;
        start_time = boost::posix_time::microsec_clock::local_time();
        previous_time = start_time;
      } else {
        // imagematch scores
        total_bytes += frame
                           ->GetValue<std::vector<std::pair<int, float>>>(
                               "imagematch.scores")
                           .size() *
                       sizeof(std::pair<int, float>);
        // vishash
        cv::Mat activations = frame->GetValue<cv::Mat>("activations");
        // total_bytes += activations.total() * sizeof(float);
        // frame id
        total_bytes += sizeof(unsigned long);
        boost::posix_time::ptime current_time =
            boost::posix_time::microsec_clock::local_time();
        double net_bw_bps =
            total_bytes * 8 / (current_time - start_time).total_seconds();
        auto fps = reader->GetHistoricalFps();
        long latency_micros =
            (current_time -
             frame->GetValue<boost::posix_time::ptime>("capture_time_micros"))
                .total_microseconds();
        if ((current_time - previous_time).total_seconds() >= 2) {
          previous_time = current_time;
          std::cout << "Level " << idx << " - Network bandwidth: " << net_bw_bps
                    << " bps , " << fps << " fps , Latency: " << latency_micros
                    << " us " << std::endl;
        }
        log_msg << net_bw_bps << "," << fps << "," << latency_micros
                << std::endl;
      }
    }
  }
  reader->UnSubscribe();

  std::ostringstream log_filepath;
  log_filepath << output_dir << "/ff_" << idx << "_"
               << boost::posix_time::to_iso_extended_string(log_time) << ".csv";
  std::ofstream log_file(log_filepath.str());
  log_file << "#network bandwidth (bps), fps, e2e latency (us)" << std::endl
           << log_msg.str();
  log_file.close();
}

void Run(const std::string& ff_conf, bool block, size_t queue_size,
         const std::string& camera_name, int input_fps, unsigned int tokens,
         const std::string& model, const std::string& layer,
         size_t nne_batch_size, const std::string& output_dir) {
  boost::posix_time::ptime log_time =
      boost::posix_time::microsec_clock::local_time();
  // Parse the ff_conf file.
  bool on_first_line = true;
  int first_im_num_queries = 0;
  std::vector<std::pair<float, size_t>> buf_params;
  std::vector<int> nums_queries;
  std::ifstream ff_conf_file(ff_conf);
  std::string line;
  unsigned int level_counter = 0;
  while (std::getline(ff_conf_file, line)) {
    std::vector<std::string> args = SplitString(line, ",");
    if (StartsWith(line, "#")) {
      // Ignore comment lines.
      continue;
    }
    CHECK(args.size() == 3)
        << "Each line of the ff_conf file must contain three items: "
        << "Selectivity,BufferLength,NumQueries";

    float sel = std::stof(args.at(0));
    unsigned long buf_len = std::stoul(args.at(1));
    int num_queries = std::stoi(args.at(2));

    std::ostringstream msg;
    msg << "Level " << level_counter << " - ";
    if (on_first_line) {
      first_im_num_queries = num_queries;
      on_first_line = false;
    } else {
      buf_params.push_back({sel, (size_t)buf_len});
      nums_queries.push_back(num_queries);
      msg << "selectivity: " << sel << ", buffer length: " << buf_len << ", ";
    }
    msg << "number of queries: " << num_queries;
    LOG(INFO) << msg.str();
    ++level_counter;
  }

  std::vector<std::shared_ptr<Processor>> procs;
  std::vector<std::thread> logger_threads;

  // Create Camera.
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  camera->SetBlockOnPush(false);
  procs.push_back(camera);

  // Create frames directory and FrameWriter.
  std::string frames_dir = output_dir + "frames/";
  boost::filesystem::path frames_dir_path(frames_dir);
  boost::filesystem::create_directory(frames_dir_path);
  std::unordered_set<std::string> fields;
  auto writer = std::make_shared<FrameWriter>(fields, frames_dir,
                                              FrameWriter::FileFormat::BINARY);
  writer->SetSource(camera->GetStream());
  writer->SetBlockOnPush(block);
  procs.push_back(writer);

  // Create Throttler.
  auto throttler = std::make_shared<Throttler>(input_fps);
  throttler->SetSource(camera->GetStream());
  throttler->SetBlockOnPush(block);
  procs.push_back(throttler);

  // Create FlowControlEntrance.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(tokens);
  fc_entrance->SetSource(throttler->GetSink());
  fc_entrance->SetBlockOnPush(block);
  procs.push_back(fc_entrance);

  // Create ImageTransformer.
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource(fc_entrance->GetSink());
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create NeuralNetEvaluator.
  std::vector<std::string> output_layer_names = {layer};
  auto nne = std::make_shared<NeuralNetEvaluator>(
      model_desc, input_shape, nne_batch_size, output_layer_names);
  nne->SetSource(transformer->GetSink());
  nne->SetBlockOnPush(block);
  procs.push_back(nne);
  auto nne_stream = nne->GetSink(layer);

  // Create ImageMatch level 0. Use the same batch size as the
  // NeuralNetEvaluator.
  auto im_0 = std::make_shared<ImageMatch>("", false, nne_batch_size);
  im_0->SetSource(nne_stream);
  im_0->SetBlockOnPush(block);
  im_0->SetQueryMatrix(first_im_num_queries, 1, 1024);
  procs.push_back(im_0);

  // FlowControlExit
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(im_0->GetSink());
  fc_exit->SetBlockOnPush(block);
  procs.push_back(fc_exit);

  // Create Logger thread for first level.
  StreamPtr fc_exit_sink = fc_exit->GetSink();
  logger_threads.push_back(std::thread([fc_exit_sink, log_time, output_dir] {
    Logger(0, fc_exit_sink, log_time, output_dir);
  }));

  // Create additional keyframe detector + ImageMatch levels in the hierarchy.
  // the hierarchy.
  StreamPtr highest_stream;
  StreamPtr kd_input_stream = nne_stream;
  for (decltype(nums_queries.size()) i = 0; i < nums_queries.size(); ++i) {
    // Create keyframe detector.
    std::pair<float, size_t> kd_buf_params = buf_params.at(i);
    std::vector<std::pair<float, size_t>> kd_buf_params_vec = {kd_buf_params};
    auto kd = std::make_shared<KeyframeDetector>(kd_buf_params_vec);
    kd->SetSource(kd_input_stream);
    kd->SetBlockOnPush(block);
    procs.push_back(kd);
    kd_input_stream = kd->GetSink(kd->GetSinkName(0));

    // Create ImageMatch.
    unsigned int kd_batch_size =
        ceil(kd_buf_params.first * kd_buf_params.second);
    auto additional_im = std::make_shared<ImageMatch>("", false, kd_batch_size);
    additional_im->SetSource(kd->GetSink(kd->GetSinkName(0)));
    additional_im->SetBlockOnPush(block);
    additional_im->SetQueryMatrix(nums_queries.at(i), 1, 1024);
    procs.push_back(additional_im);

    StreamPtr additional_im_sink = additional_im->GetSink();
    logger_threads.push_back(
        std::thread([i, additional_im_sink, log_time, output_dir] {
          Logger(i + 1, additional_im_sink, log_time, output_dir);
        }));
    if (i == nums_queries.size() - 1) {
      highest_stream = additional_im->GetSink();
    }
  }

  // Launch stopped thread.
  std::thread stopper_thread([highest_stream] { Stopper(highest_stream); });

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start(queue_size);
  }

  while (!stopped) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }

  stopper_thread.join();
  for (auto& t : logger_threads) {
    t.join();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("FilterForward");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()(
      "ff-conf,f", po::value<std::string>(),
      "The file containing the keyframe detector's configuration.");
  desc.add_options()("block,b", "Processors should block when pushing frames.");
  desc.add_options()("queue-size,q", po::value<size_t>()->default_value(16),
                     "queue size");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("input-fps,i", po::value<int>()->default_value(30),
                     "input fps");
  desc.add_options()("tokens,t", po::value<unsigned int>()->default_value(5),
                     "The number of flow control tokens to issue.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("layer,l", po::value<std::string>()->required(),
                     "The layer to extract and use as the basis for keyframe "
                     "detection and imagematch(tm)");
  desc.add_options()("nne-batch-size,s", po::value<size_t>()->default_value(1),
                     "nne batch size");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory where we'll write files.");
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
  bool block = args.count("block");
  size_t queue_size = args["queue-size"].as<size_t>();
  std::string camera = args["camera"].as<std::string>();
  int input_fps = args["input-fps"].as<int>();
  unsigned int tokens = args["tokens"].as<unsigned int>();
  std::string layer = args["layer"].as<std::string>();
  std::string model = args["model"].as<std::string>();
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  std::string output_dir = args["output-dir"].as<std::string>();
  Run(ff_conf, block, queue_size, camera, input_fps, tokens, model, layer,
      nne_batch_size, output_dir);
  return 0;
}
