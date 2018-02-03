
#include <atomic>
#include <climits>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
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
#include "processor/jpeg_writer.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"
#include "processor/temporal_region_selector.h"
#include "stream/frame.h"
#include "stream/stream.h"

namespace po = boost::program_options;

constexpr unsigned long MIN_START_ID = 0;
constexpr unsigned long MAX_END_ID = ULONG_MAX;
constexpr auto FAKE_FV_KEY = "fv";

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

// This function feeds fake vishashes to the specified stream.
void Feeder(size_t fake_vishash_length, const std::string& fv_key,
            unsigned long num_frames, StreamPtr vishash_stream) {
  while (!started) {
    LOG(INFO) << "Waiting to start...";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  unsigned long frame_id = 0;
  for (; frame_id < num_frames; ++frame_id) {
    auto frame = std::make_unique<Frame>();
    frame->SetValue("frame_id", frame_id);

    cv::Mat vishash(fake_vishash_length, 1, CV_32FC1);
    vishash.at<float>(0, 0) = (float)frame_id;
    frame->SetValue(fv_key, vishash);

    vishash_stream->PushFrame(std::move(frame), true);
  }

  // We need to push a stop frame in order to signal the pipeline (and
  // the rest of the application logic) to stop.
  auto stop_frame = std::make_unique<Frame>();
  stop_frame->SetValue("frame_id", frame_id);
  stop_frame->SetStopFrame(true);
  vishash_stream->PushFrame(std::move(stop_frame), true);
}

int NumFramesPerTopLevelRun(std::vector<std::pair<float, size_t>> buf_params,
                            int end_idx) {
  if (end_idx == 0) {
    return buf_params.at(0).second;
  }
  size_t buf_len = buf_params.at(end_idx).second;
  size_t buf_len_prev = buf_params.at(end_idx - 1).second;
  float sel_prev = buf_params.at(end_idx - 1).first;
  return buf_len / (buf_len_prev * sel_prev) *
         NumFramesPerTopLevelRun(buf_params, end_idx - 1);
}

void Run(const std::string& kd_conf, size_t queue_size, bool block,
         unsigned long start_id, unsigned long end_id, unsigned int num_frames,
         bool generate_fake_vishashes, size_t fake_vishash_length,
         const std::string& camera_name, const std::string& model,
         const std::string& layer, size_t nne_batch_size, bool save_jpegs,
         const std::string& output_dir) {
  std::vector<std::pair<float, size_t>> buf_params;
  unsigned int levels = 0;
  std::ifstream kd_conf_file(kd_conf);
  std::string line;
  while (std::getline(kd_conf_file, line)) {
    std::vector<std::string> args = SplitString(line, ",");
    if (StartsWith(line, "#")) {
      // Ignore comment lines.
      continue;
    }
    CHECK(args.size() == 2)
        << "Each line of the ff_conf file must contain two items: "
        << "Selectivity,BufferLength";

    float sel = std::stof(args.at(0));
    unsigned long buf_len = std::stoul(args.at(1));
    buf_params.push_back({sel, (size_t)buf_len});

    std::ostringstream msg;
    msg << "Level " << levels << " - "
        << "selectivity: " << sel << ", buffer length: " << buf_len;
    LOG(INFO) << msg.str();
    ++levels;
  }

  std::vector<std::shared_ptr<Processor>> procs;
  // This stream will contain layer activations to feed into the keyframe
  // detector.
  StreamPtr vishash_stream;
  // This thread will generate fake layer activations, if the app has been told
  // to do so.
  std::thread feeder;

  std::string fv_key;
  if (generate_fake_vishashes) {
    LOG(INFO) << "Generating fake vishashes";
    fv_key = FAKE_FV_KEY;
    // Instead of reading frames from a camera or a file, we will generate fake
    // layer activations. This enables rapid evaluation of the keyframe
    // detector's performance because it eliminates the overhead of running a
    // DNN.
    vishash_stream = StreamPtr(new Stream());
    feeder =
        std::thread([fake_vishash_length, fv_key, num_frames, vishash_stream] {
          Feeder(fake_vishash_length, fv_key, num_frames, vishash_stream);
        });
  } else {
    fv_key = layer;

    // Create Camera.
    auto camera = CameraManager::GetInstance().GetCamera(camera_name);
    camera->SetBlockOnPush(block);
    procs.push_back(camera);

    StreamPtr stream_to_transform;
    if (start_id != MIN_START_ID || end_id != MAX_END_ID) {
      // If we're supposed to select a temporal subset of the frames, then
      // we need to create a TemporalRegionSelector.
      auto selector =
          std::make_shared<TemporalRegionSelector>(start_id, end_id);
      selector->SetSource("input", camera->GetStream());
      selector->SetBlockOnPush(block);
      procs.push_back(selector);
      stream_to_transform = selector->GetSink("output");
    } else {
      stream_to_transform = camera->GetStream();
    }

    // Create ImageTransformer.
    ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
    Shape input_shape(3, model_desc.GetInputWidth(),
                      model_desc.GetInputHeight());
    auto transformer =
        std::make_shared<ImageTransformer>(input_shape, true, true);
    transformer->SetSource("input", stream_to_transform);
    transformer->SetBlockOnPush(block);
    procs.push_back(transformer);

    // Create NeuralNetEvaluator.
    std::vector<std::string> output_layer_names = {layer};
    auto nne = std::make_shared<NeuralNetEvaluator>(
        model_desc, input_shape, nne_batch_size, output_layer_names);
    nne->SetSource(transformer->GetSink("output"));
    nne->SetBlockOnPush(block);
    procs.push_back(nne);
    vishash_stream = nne->GetSink(layer);
  }

  // Create KeyframeDetector.
  auto kd = std::make_shared<KeyframeDetector>(fv_key, buf_params);
  kd->SetSource(vishash_stream);
  kd->SetBlockOnPush(block);
  kd->EnableLog(output_dir);
  procs.push_back(kd);

  StreamPtr kd_stream = kd->GetSink("output_0");
  if (save_jpegs) {
    // Create JpegWriter.
    auto writer = std::make_shared<JpegWriter>("original_image", output_dir);
    writer->SetSource(kd_stream);
    procs.push_back(writer);
  }

  // Subscribe before starting the processors so that we definitely do not miss
  // any frames.
  StreamReader* reader = kd_stream->Subscribe();

  // Start the Processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start(queue_size);
  }

  // Signal the feeder thread to start. If there is no feeder thread, then this
  // does nothing.
  started = true;

  LOG(INFO) << "Num frames per top level run: "
            << NumFramesPerTopLevelRun(buf_params, buf_params.size() - 1);

  std::ofstream micros_log(output_dir + "/kd_micros.txt");
  while (true) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      if (frame->IsStopFrame()) {
        break;
      }

      std::ostringstream time_key;
      time_key << "kd_level_" << levels << "_micros";
      std::string time_key_str = time_key.str();
      if (frame->Count(time_key_str)) {
        micros_log << frame
                          ->GetValue<boost::posix_time::time_duration>(
                              time_key_str)
                          .total_microseconds()
                   << "\n";
      }
    }
  }
  micros_log.close();

  if (feeder.joinable()) {
    vishash_stream->Stop();
    feeder.join();
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("KeyframeDetector demo");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config-dir", po::value<std::string>(),
                     "The directory which contains Streamer's configuration "
                     "files");
  desc.add_options()("kd-conf", po::value<std::string>()->required(),
                     "The file in which keyframe detector configuration params "
                     "are stored.");
  desc.add_options()("queue-size", po::value<size_t>()->default_value(16),
                     "The size of the queues between processors.");
  desc.add_options()("block", "Processors should block when pushing frames.");
  desc.add_options()("start-frame",
                     po::value<unsigned long>()->default_value(MIN_START_ID),
                     "The frame id (starting from 0) of the first frame to "
                     "process.");
  desc.add_options()("end-frame",
                     po::value<unsigned long>()->default_value(MAX_END_ID),
                     "The frame id (starting from 0) of the last frame to "
                     "process.");
  desc.add_options()("num-frames",
                     po::value<unsigned int>()->default_value(1000),
                     "The number of frames to create when generating fake "
                     "vishashes. Must be used in conjunction with "
                     "\"--fake-vishashes\" and \"--fake-vishash-length\".");
  desc.add_options()("fake-vishashes",
                     "Generate fake vishashes. Must be used in conjunction "
                     "with \"--fake-vishash-length\" and \"--num-frames\".");
  desc.add_options()("fake-vishash-length",
                     po::value<size_t>()->default_value(1024),
                     "The number of values in the fake vishashes that will be "
                     "generated. Must be used in conjunction with "
                     "\"--fake-vishashes\" and \"--num-frames\".");
  desc.add_options()("camera", po::value<std::string>(),
                     "The name of the camera to use");
  desc.add_options()("model", po::value<std::string>(),
                     "The name of the model to run");
  desc.add_options()("layer", po::value<std::string>(),
                     "The layer to extract and use as the basis for keyframe "
                     "detection");
  desc.add_options()("nne-batch-size,s", po::value<size_t>()->default_value(1),
                     "Batch size of the NeuralNetEvaluator.");
  desc.add_options()("save-jpegs",
                     "Save a JPEG of each keyframe in \"--output-dir\"");
  desc.add_options()("output-dir", po::value<std::string>()->required(),
                     "The directory in which to store the keyframe JPEGs.");

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

  std::string kd_conf = args["kd-conf"].as<std::string>();
  size_t queue_size = args["queue-size"].as<size_t>();
  bool block = args.count("block");
  unsigned int num_frames = args["num-frames"].as<unsigned int>();
  unsigned long start_id = args["start-frame"].as<unsigned long>();
  unsigned long end_id = args["end-frame"].as<unsigned long>();
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
  bool save_jpegs = args.count("save-jpegs");
  std::string output_dir = args["output-dir"].as<std::string>();

  if (generate_fake_vishashes) {
    if (save_jpegs) {
      std::cerr << "\"--save-jpegs\" is incompatible with \"--fake-vishashes\"!"
                << std::endl;
      return 1;
    }
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

  Run(kd_conf, queue_size, block, start_id, end_id, num_frames,
      generate_fake_vishashes, fake_vishash_length, camera, model, layer,
      nne_batch_size, save_jpegs, output_dir);
  return 0;
}
