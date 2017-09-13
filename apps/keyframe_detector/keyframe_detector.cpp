
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
void Feeder(size_t fake_vishash_length, unsigned long num_frames,
            StreamPtr vishash_stream) {
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
    frame->SetValue("activations", vishash);

    vishash_stream->PushFrame(std::move(frame), true);
  }

  // We need to push a stop frame in order to signal the pipeline (and
  // the rest of the application logic) to stop.
  auto stop_frame = std::make_unique<Frame>();
  stop_frame->SetValue("frame_id", frame_id);
  stop_frame->SetStopFrame(true);
  vishash_stream->PushFrame(std::move(stop_frame), true);
}

void Run(const std::string& camera_name, const std::string& model,
         const std::string& layer, float sel, size_t buf_len, size_t levels,
         const std::string& output_dir, bool save_jpegs, bool block,
         bool generate_fake_vishashes, size_t fake_vishash_length,
         unsigned long num_frames, unsigned long start_id,
         unsigned long end_id) {
  std::vector<std::shared_ptr<Processor>> procs;
  // This stream will contain layer activations to feed into the keyframe
  // detector.
  StreamPtr vishash_stream;
  // This thread will generate fake layer activations, if the app has been told
  // to do so.
  std::thread feeder;

  if (generate_fake_vishashes) {
    LOG(INFO) << "Generating fake vishashes";
    // Instead of reading frames from a camera or a file, we will generate fake
    // layer activations. This enables rapid evaluation of the keyframe
    // detector's performance because it eliminates the overhead of running a
    // DNN.
    vishash_stream = StreamPtr(new Stream());
    feeder = std::thread([fake_vishash_length, num_frames, vishash_stream] {
      Feeder(fake_vishash_length, num_frames, vishash_stream);
    });
  } else {
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
    auto model_desc = ModelManager::GetInstance().GetModelDesc(model);
    Shape input_shape(3, model_desc.GetInputWidth(),
                      model_desc.GetInputHeight());
    auto transformer =
        std::make_shared<ImageTransformer>(input_shape, true, true);
    transformer->SetSource("input", stream_to_transform);
    transformer->SetBlockOnPush(block);
    procs.push_back(transformer);

    // Create NeuralNetEvaluator.
    std::vector<std::string> output_layer_names = {layer};
    auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                    output_layer_names);
    nne->SetSource(transformer->GetSink("output"));
    nne->SetBlockOnPush(block);
    procs.push_back(nne);
    vishash_stream = nne->GetSink(layer);
  }

  // Create KeyframeDetector.
  std::vector<std::pair<float, size_t>> buf_params(levels, {sel, buf_len});
  auto kd = std::make_shared<KeyframeDetector>(buf_params);
  kd->SetSource(vishash_stream);
  kd->SetBlockOnPush(block);
  kd->EnableLog(output_dir);
  procs.push_back(kd);

  auto kd_stream = kd->GetSink("output_0");
  if (save_jpegs) {
    // Create JpegWriter.
    auto writer = std::make_shared<JpegWriter>("original_image", output_dir);
    writer->SetSource(kd_stream);
    procs.push_back(writer);
  }

  // Subscribe before starting the processors so that we definitely do not miss
  // any frames.
  auto reader = kd_stream->Subscribe();

  // Start the Processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  // Signal the feeder thread to start. If there is no feeder thread, then this
  // does nothing.
  started = true;

  std::ofstream micros_log(output_dir + "/kd_micros.txt");
  while (true) {
    auto frame = reader->PopFrame();
    if (frame->IsStopFrame()) {
      break;
    }

    std::ostringstream time_key;
    time_key << "kd_level_0_micros";
    auto time_key_str = time_key.str();
    if (frame->Count(time_key_str)) {
      micros_log << frame->GetValue<long>(time_key_str) << "\n";
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
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory which contains Streamer's configuration "
                     "files");
  desc.add_options()("camera,c", po::value<std::string>(),
                     "The name of the camera to use");
  desc.add_options()("model,m", po::value<std::string>(),
                     "The name of the model to run");
  desc.add_options()("layer,l", po::value<std::string>(),
                     "The layer to extract and use as the basis for keyframe "
                     "detection");
  desc.add_options()("sel,s", po::value<float>()->default_value(0.1),
                     "The selectivity to use, in the range (0, 1]");
  desc.add_options()("buf-len,b", po::value<size_t>()->default_value(1000),
                     "The number of frames to buffer before detecting "
                     "keyframes.");
  desc.add_options()("levels,v", po::value<size_t>()->default_value(1),
                     "The number of hierarchy levels to create.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to store the keyframe JPEGs.");
  desc.add_options()("save-jpegs,j",
                     "Save a JPEG of each keyframe in \"--output-dir\"");
  desc.add_options()("block-on-push,p",
                     "Processors should block when pushing frames.");
  desc.add_options()("fake-vishashes",
                     "Generate fake vishashes. Must be used in conjunction "
                     "with \"--fake-vishash-length\" and \"--num-frames\".");
  desc.add_options()("fake-vishash-length",
                     po::value<size_t>()->default_value(1024),
                     "The number of values in the fake vishashes that will be "
                     "generated. Must be used in conjunction with "
                     "\"--fake-vishashes\" and \"--num-frames\".");
  desc.add_options()("num-frames",
                     po::value<unsigned long>()->default_value(1000),
                     "The number of frames to create when generating fake "
                     "vishashes. Must be used in conjunction with "
                     "\"--fake-vishashes\" and \"--fake-vishash-length\".");
  desc.add_options()("start-frame",
                     po::value<unsigned long>()->default_value(MIN_START_ID),
                     "The frame id (starting from 0) of the first frame to "
                     "process.");
  desc.add_options()("end-frame",
                     po::value<unsigned long>()->default_value(MAX_END_ID),
                     "The frame id (starting from 0) of the last frame to "
                     "process.");

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
  float sel = args["sel"].as<float>();
  size_t buf_len = args["buf-len"].as<size_t>();
  size_t levels = args["levels"].as<size_t>();
  std::string output_dir = args["output-dir"].as<std::string>();
  bool save_jpegs = args.count("save-jpegs");
  bool block = args.count("block-on-push");
  bool generate_fake_vishashes = args.count("fake-vishashes");
  size_t fake_vishash_length = args["fake-vishash-length"].as<size_t>();
  unsigned long num_frames = args["num-frames"].as<unsigned long>();
  unsigned long start_id = args["start-frame"].as<unsigned long>();
  unsigned long end_id = args["end-frame"].as<unsigned long>();

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

  Run(camera, model, layer, sel, buf_len, levels, output_dir, save_jpegs, block,
      generate_fake_vishashes, fake_vishash_length, num_frames, start_id,
      end_id);
  return 0;
}
