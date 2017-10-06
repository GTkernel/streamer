
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "processor/frame_writer.h"
#include "processor/image_transformer.h"
#include "processor/jpeg_writer.h"
#include "processor/neural_net_evaluator.h"
#include "stream/frame.h"

namespace po = boost::program_options;

// Whether the pipeline has been stopped.
std::atomic<bool> stopped(false);

void ProgressTracker(StreamPtr stream) {
  StreamReader* reader = stream->Subscribe();
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      std::cout << "\rSaved feature vector for frame "
                << frame->GetValue<unsigned long>("frame_id");
      // This is required in order to make the console update as soon as the
      // above log is printed. Without this, the progress log will not update
      // smoothly.
      std::cout.flush();
    }
  }
  reader->UnSubscribe();
}

void Run(const std::string& camera_name, const std::string& model_name,
         const std::string& layer, bool block, const std::string& output_dir,
         unsigned long frames_per_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);
  StreamPtr camera_stream = camera->GetStream();

  // Create JpegWriter.
  auto jpeg_writer = std::make_shared<JpegWriter>("original_image", output_dir,
                                                  false, frames_per_dir);
  jpeg_writer->SetSource(camera_stream);
  jpeg_writer->SetBlockOnPush(block);
  procs.push_back(jpeg_writer);

  // Create ImageTransformer.
  ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model_name);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource(camera_stream);
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create NeuralNetEvaluator.
  auto nne = std::make_shared<NeuralNetEvaluator>(
      model_desc, input_shape, 4, std::vector<std::string>{layer});
  nne->SetSource(transformer->GetSink(), layer);
  nne->SetBlockOnPush(block);
  procs.push_back(nne);
  StreamPtr nne_stream = nne->GetSink(layer);

  // Create FrameWriter.
  auto frame_writer = std::make_shared<FrameWriter>(
      std::unordered_set<std::string>{"frame_id", "activations"}, output_dir,
      FrameWriter::FileFormat::JSON, false, false, frames_per_dir);
  frame_writer->SetSource(nne_stream);
  frame_writer->SetBlockOnPush(block);
  procs.push_back(frame_writer);

  auto reader = nne->GetSink(layer)->Subscribe();

  std::thread progress_thread =
      std::thread([nne_stream] { ProgressTracker(nne_stream); });

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  // Loop until we receive a stop frame.
  while (true) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr && frame->IsStopFrame()) {
      break;
    }
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }

  // Signal the progress thread to stop.
  stopped = true;
  progress_thread.join();
}

int main(int argc, char* argv[]) {
  po::options_description desc(
      "Stores the features vector for a camera stream as text files.");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()(
      "config-dir,C", po::value<std::string>(),
      "The directory containing streamer's configuration files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("layer,l", po::value<std::string>()->required(),
                     "The name of the layer to extract.");
  desc.add_options()("block,b", "Whether to block when pushing frames.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to store the JPEGs and vishash "
                     "file.");
  desc.add_options()("frames-per-dir,n",
                     po::value<unsigned long>()->default_value(1000),
                     "The number of frames to put in each output subdir.");

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

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = args["camera"].as<std::string>();
  auto model = args["model"].as<std::string>();
  auto layer = args["layer"].as<std::string>();
  bool block = args.count("block");
  auto output_dir = args["output-dir"].as<std::string>();
  auto frames_per_dir = args["frames-per-dir"].as<unsigned long>();
  Run(camera_name, model, layer, block, output_dir, frames_per_dir);
  return 0;
}
