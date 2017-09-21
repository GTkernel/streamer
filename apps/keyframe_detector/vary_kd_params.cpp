
#include <atomic>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "processor/image_transformer.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"
#include "stream/frame.h"
#include "stream/stream.h"

namespace po = boost::program_options;

// Set to true when the pipeline has been started. Used to signal the feeder
// thread to start, if it exists.
std::atomic<bool> started(false);

void ErrorRequired(std::string param) {
  std::cerr << "\"--" << param << "\" required!" << std::endl;
}

void WarnUnused(std::string param) {
  LOG(WARNING) << "\"--" << param
               << "\" ignored when using \"--fake-vishashes\"!";
}

void UpdateProgressBar(double current_step, double total_steps,
                       const std::string& msg = "") {
  std::ostringstream progress_msg;
  progress_msg << "[";
  double i = 0;
  double current_percent = (current_step / total_steps) * 100;
  for (; i < current_percent / 2; ++i) {
    progress_msg << "|";
  }
  for (; i < 50; ++i) {
    progress_msg << "-";
  }
  progress_msg << "] " << std::setprecision(3) << current_percent << "% - "
               << msg;

  // Include to spaces at the end to make sure that the previous line is always
  // overridden.
  std::cout << "\r" << progress_msg.str() << "  ";
  // This is required in order to make the console update as soon as the above
  // log is printed. Without this, the progress log will not update smoothly.
  std::cout.flush();
}

// Feeds frames to the specified stream.
void Feeder(std::shared_ptr<std::vector<std::unique_ptr<Frame>>> frames,
            StreamPtr stream, bool block) {
  while (!started) {
    LOG(INFO) << "Feeder waiting to start...";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  for (auto& frame : *frames) {
    stream->PushFrame(std::make_unique<Frame>(frame), block);
  }
}

// Generates vishashes, either randomly or by passing camera frames through a
// DNN.
std::shared_ptr<std::vector<std::unique_ptr<Frame>>> GenerateVishashes(
    size_t queue_size, bool block, unsigned long num_frames,
    bool generate_fake_vishashes, size_t fake_vishash_length,
    const std::string& camera_name, const std::string& model,
    const std::string& layer, size_t nne_batch_size) {
  auto frames = std::make_shared<std::vector<std::unique_ptr<Frame>>>();

  if (generate_fake_vishashes) {
    for (unsigned long frame_id = 0; frame_id < num_frames; ++frame_id) {
      UpdateProgressBar(frame_id + 1, num_frames, "Generating vishashes...");

      auto frame = std::make_unique<Frame>();
      frame->SetValue("frame_id", frame_id);

      cv::Mat vishash(fake_vishash_length, 1, CV_32FC1);
      cv::randu(vishash, 0, 1);
      frame->SetValue("activations", vishash);

      frames->push_back(std::move(frame));
    }
    std::cout << std::endl;
  } else {
    std::vector<std::shared_ptr<Processor>> procs;

    // Create Camera.
    auto camera = CameraManager::GetInstance().GetCamera(camera_name);
    camera->SetBlockOnPush(block);
    procs.push_back(camera);

    // Create ImageTransformer.
    ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
    Shape input_shape(3, model_desc.GetInputWidth(),
                      model_desc.GetInputHeight());
    auto transformer =
        std::make_shared<ImageTransformer>(input_shape, true, true);
    transformer->SetSource(camera->GetStream());
    transformer->SetBlockOnPush(block);
    procs.push_back(transformer);

    // Create NeuralNetEvaluator.
    std::vector<std::string> output_layer_names = {layer};
    auto nne = std::make_shared<NeuralNetEvaluator>(
        model_desc, input_shape, nne_batch_size, output_layer_names);
    nne->SetSource(transformer->GetSink());
    nne->SetBlockOnPush(block);
    procs.push_back(nne);

    // Subscribe before starting the processors so that we definitely do not
    // miss any frames.
    StreamReader* reader = nne->GetSink(layer)->Subscribe();

    // Start the Processors in reverse order.
    for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
      (*procs_it)->Start(queue_size);
    }

    // Loop until either we have received "num_frames" frames or we receive a
    // stop frame.
    unsigned long count = 0;
    while (count < num_frames) {
      if (num_frames != ULONG_MAX) {
        UpdateProgressBar(count + 1, num_frames, "Generating vishashes...");
      }

      std::unique_ptr<Frame> frame = reader->PopFrame();
      if (frame != nullptr) {
        if (frame->IsStopFrame()) {
          break;
        } else {
          frames->push_back(std::move(frame));
          ++count;
        }
      }
    }
    std::cout << std::endl;
    reader->UnSubscribe();

    // Stop the processors in forward order.
    for (const auto& proc : procs) {
      proc->Stop();
    }
  }

  // Create a stop frame to signal the end of the frame stream.
  auto stop_frame = std::make_unique<Frame>();
  stop_frame->SetStopFrame(true);
  stop_frame->SetValue("frame_id",
                       frames->back()->GetValue<unsigned long>("frame_id") + 1);
  frames->push_back(std::move(stop_frame));
  return frames;
}

void RunKeyframeDetector(
    std::shared_ptr<std::vector<std::unique_ptr<Frame>>> frames,
    size_t queue_size, bool block, float sel,

    size_t buf_len, size_t levels, const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;
  StreamPtr frame_stream = std::make_shared<Stream>();

  // Create Feeder Thread.
  std::thread feeder(
      [frames, frame_stream, block] { Feeder(frames, frame_stream, block); });

  // Create KeyframeDetector. Each buffer has the same selectivity and length.
  std::vector<std::pair<float, size_t>> buf_params(
      levels, std::pair<float, size_t>{sel, buf_len});
  auto kd = std::make_shared<KeyframeDetector>(buf_params);
  kd->SetSource(frame_stream);
  kd->SetBlockOnPush(block);
  kd->EnableLog(output_dir);
  procs.push_back(kd);

  // Subscribe before starting the processors so that we definitely do not miss
  // any frames.
  StreamReader* reader = kd->GetSink("output_0")->Subscribe();

  // Start the Processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start(queue_size);
  }

  // Signal the feeder thread to start.
  started = true;

  // Create a log file for the running time of the keyframe detection algorithm.
  std::ostringstream micros_filepath;
  micros_filepath << output_dir << "/keyframe_detector_" << sel << "_"
                  << buf_len << "_" << levels << "_micros.txt";
  std::ofstream micros_log(micros_filepath.str());

  while (true) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      if (frame->IsStopFrame()) {
        break;
      }

      std::ostringstream time_key;
      time_key << "kd_level_0_micros";
      std::string time_key_str = time_key.str();
      if (frame->Count(time_key_str)) {
        micros_log << frame->GetValue<long>(time_key_str) << std::endl;
      }
    }
  }
  micros_log.close();
  reader->UnSubscribe();

  if (feeder.joinable()) {
    frame_stream->Stop();
    feeder.join();
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

void Run(size_t queue_size, bool block, unsigned long num_frames,
         bool generate_fake_vishashes, size_t fake_vishash_length,
         const std::string& camera_name, const std::string& model,
         const std::string& layer, size_t nne_batch_size,
         std::vector<float> sels, std::vector<size_t> buf_lens,
         std::vector<size_t> nums_levels, const std::string& output_dir) {
  std::shared_ptr<std::vector<std::unique_ptr<Frame>>> frames =
      GenerateVishashes(queue_size, block, num_frames, generate_fake_vishashes,
                        fake_vishash_length, camera_name, model, layer,
                        nne_batch_size);

  std::cout << "Running keyframe detector experiments..." << std::endl;

  double total_steps = sels.size() * buf_lens.size() * nums_levels.size();
  double current_step = 0;
  for (auto sel : sels) {
    for (auto buf_len : buf_lens) {
      for (auto levels : nums_levels) {
        // Log progress.
        std::ostringstream msg;
        msg << "Current experiment: Selectivity: " << sel
            << " , Buffer Length: " << buf_len << " , Levels: " << levels;
        UpdateProgressBar(++current_step, total_steps, msg.str());

        // Create a subdirectory for this experiment's results.
        std::ostringstream output_subdir;
        output_subdir << output_dir << "/" << sel << "_" << buf_len << "_"
                      << levels;
        std::string output_subdir_str = output_subdir.str();
        boost::filesystem::path output_subdir_path(output_subdir_str);
        boost::filesystem::create_directory(output_subdir_path);

        RunKeyframeDetector(frames, queue_size, block, sel, buf_len, levels,
                            output_subdir_str);
      }
    }
  }

  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  po::options_description desc("Vary KeyframeDetector parameters.");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config-dir", po::value<std::string>(),
                     "The directory which contains Streamer's configuration "
                     "files");
  desc.add_options()("queue-size", po::value<size_t>()->default_value(16),
                     "The size of the queues between processors.");
  desc.add_options()("block", "Processors should block when pushing frames.");
  desc.add_options()("num-frames",
                     po::value<unsigned long>()->default_value(ULONG_MAX),
                     "The number of frames to create when generating fake "
                     "vishashes. If this is used in conjunction with a camera "
                     "stream and the camera stream contains fewer than "
                     "\"--num-frames\" frames, then this option is ignored.");
  desc.add_options()("fake-vishashes",
                     "Generate fake vishashes. Must be used in conjunction "
                     "with \"--fake-vishash-length\".");
  desc.add_options()("fake-vishash-length",
                     po::value<size_t>()->default_value(1024),
                     "The number of values in the fake vishashes that will be "
                     "generated. Should be used in conjunction with "
                     "\"--fake-vishashes\".");
  desc.add_options()("camera", po::value<std::string>(),
                     "The name of the camera to use.");
  desc.add_options()("model", po::value<std::string>(),
                     "The name of the model to use.");
  desc.add_options()("layer", po::value<std::string>(),
                     "The layer to extract and use as the basis for keyframe "
                     "detection.");
  desc.add_options()("nne-batch-size", po::value<size_t>()->default_value(1),
                     "The batch size to use during DNN evaluation.");
  desc.add_options()(
      "sels",
      po::value<std::vector<float>>()->multitoken()->composing()->required(),
      "The selectivities to use. Designed to be specified multiple times.");
  desc.add_options()(
      "buf-lens",
      po::value<std::vector<size_t>>()->multitoken()->composing()->required(),
      "The buffer lengths to use. Designed to be specified multiple times.");
  desc.add_options()(
      "levels",
      po::value<std::vector<size_t>>()->multitoken()->composing()->required(),
      "The numbers of levels to use. Designed to be specified multiple times.");
  desc.add_options()("output-dir", po::value<std::string>()->required(),
                     "The directory in which to store output log files.");

  // Parse command line arguments.
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

  // Set up GStreamer
  gst_init(&argc, &argv);
  // Set up glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  size_t queue_size = args["queue-size"].as<size_t>();
  bool block = args.count("block");
  unsigned long num_frames = args["num-frames"].as<unsigned long>();
  bool generate_fake_vishashes = args.count("fake-vishashes");
  size_t fake_vishash_length = args["fake-vishash-length"].as<size_t>();
  std::string camera = "";
  if (args.count("camera")) {
    camera = args["camera"].as<std::string>();
  }
  std::string model = "";
  if (args.count("model")) {
    model = args["model"].as<std::string>();
  }
  std::string layer = "";
  if (args.count("layer")) {
    layer = args["layer"].as<std::string>();
  }
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  std::vector<float> sels = args["sels"].as<std::vector<float>>();
  std::vector<size_t> buf_lens = args["buf-lens"].as<std::vector<size_t>>();
  std::vector<size_t> nums_levels = args["levels"].as<std::vector<size_t>>();
  std::string output_dir = args["output-dir"].as<std::string>();
  if (generate_fake_vishashes) {
    if (camera != "") {
      WarnUnused("camera");
    }
    if (model != "") {
      WarnUnused("model");
    }
    if (layer != "") {
      WarnUnused("layer");
    }
  } else {
    if (camera == "") {
      ErrorRequired("camera");
      return 1;
    }
    if (model == "") {
      ErrorRequired("model");
      return 1;
    }
    if (layer == "") {
      ErrorRequired("layer");
      return 1;
    }
  }

  Run(queue_size, block, num_frames, generate_fake_vishashes,
      fake_vishash_length, camera, model, layer, nne_batch_size, sels, buf_lens,
      nums_levels, output_dir);
  return 0;
}
