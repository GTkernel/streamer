
#include <iostream>
#include <memory>
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
#include "processor/processor.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, const std::string& model,
         const std::string& layer, float sel, size_t buf_len, size_t levels,
         const std::string& output_dir, bool save_jpegs, bool block) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  camera->SetBlockOnPush(block);
  procs.push_back(camera);

  // Create ImageTransformer.
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource("input", camera->GetStream());
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create KeyframeDetector.
  std::vector<std::pair<float, size_t>> buf_params(levels, {sel, buf_len});
  auto kd = std::make_shared<KeyframeDetector>(model_desc, input_shape, layer,
                                               buf_params);
  kd->SetSource(transformer->GetSink("output"));
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

  // Start the Processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  auto reader = kd_stream->Subscribe();
  while (true) {
    auto frame = reader->PopFrame();
    if (frame->IsStopFrame()) {
      break;
    }
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
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to run");
  desc.add_options()("layer,l", po::value<std::string>()->required(),
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
                     "Save a JPEG of each keyframe in --output-dir");
  desc.add_options()("block-on-push,p",
                     "Processors should block when pushing frames.");

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
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  std::string camera = args["camera"].as<std::string>();
  std::string model = args["model"].as<std::string>();
  std::string layer = args["layer"].as<std::string>();
  float sel = args["sel"].as<float>();
  size_t buf_len = args["buf-len"].as<size_t>();
  size_t levels = args["levels"].as<size_t>();
  std::string output_dir = args["output-dir"].as<std::string>();
  bool save_jpegs = args.count("save-jpegs");
  bool block = args.count("block-on-push");

  Run(camera, model, layer, sel, buf_len, levels, output_dir, save_jpegs,
      block);
  return 0;
}
