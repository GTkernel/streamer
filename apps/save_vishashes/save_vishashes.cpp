
#include <cstdio>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "processor/image_transformer.h"
#include "processor/jpeg_writer.h"
#include "processor/neural_net_evaluator.h"
#include "stream/frame.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, const std::string& model_name,
         const std::string& layer, bool block, const std::string& output_dir,
         unsigned int frames_per_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Camera.
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);
  auto camera_stream = camera->GetSink("output");

  // JpegWriter.
  auto jpeg_writer = std::make_shared<JpegWriter>("original_image", output_dir,
                                                  frames_per_dir);
  jpeg_writer->SetSource(camera_stream);
  jpeg_writer->SetBlockOnPush(block);
  procs.push_back(jpeg_writer);

  // ImageTransformer.
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model_name);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource("input", camera_stream);
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // NeuralNetEvaluator.
  std::vector<std::string> output_layer_names = {layer};
  auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, 1,
                                                  output_layer_names);
  nne->SetSource("input", transformer->GetSink("output"), layer);
  nne->SetBlockOnPush(block);
  procs.push_back(nne);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  auto reader = nne->GetSink(layer)->Subscribe();
  while (true) {
    auto frame = reader->PopFrame();
    if (frame->IsStopFrame()) {
      break;
    }
    auto frame_id = frame->GetValue<unsigned long>("frame_id");
    LOG(INFO) << "Saving feature vector for frame: " << frame_id;

    // Save vishash to file.
    std::ostringstream dirpath;
    auto dir_num = frame_id / frames_per_dir;
    dirpath << output_dir << "/" << dir_num;
    auto dirpath_str = dirpath.str();
    boost::filesystem::path dir(dirpath_str);
    boost::filesystem::create_directory(dir);

    std::ostringstream filepath;
    filepath << dirpath_str << "/" << frame_id << "_feature_vector.txt";
    std::ofstream vishash_file(filepath.str());

    auto activations = frame->GetValue<cv::Mat>("activations");
    for (auto it = activations.begin<float>(); it != activations.end<float>();
         ++it) {
      vishash_file << *it << "\n";
    }
    vishash_file.close();
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Runs image classification on a video stream");
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
                     po::value<unsigned int>()->default_value(100),
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
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  auto camera_name = args["camera"].as<std::string>();
  auto model = args["model"].as<std::string>();
  auto layer = args["layer"].as<std::string>();
  bool block = args.count("block");
  auto output_dir = args["output-dir"].as<std::string>();
  auto frames_per_dir = args["frames-per-dir"].as<unsigned int>();
  Run(camera_name, model, layer, block, output_dir, frames_per_dir);
  return 0;
}
