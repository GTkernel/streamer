
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <gst/gst.h>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/processor.h"
#include "processor/throttler.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/imagematch/imagematch.h"
#include "processor/neural_net_evaluator.h"
#include "processor/frame_writer.h"
#include "processor/image_transformer.h"

namespace po = boost::program_options;

void Run(const std::string& ff_conf, 
         bool block, const std::string& camera_name,
         int input_fps, unsigned int tokens,
         const std::string& model,
         const std::string& layer,
         size_t nne_batch_size,
         const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  auto camera = CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Create frames directory and FrameWriter.
  std::string frames_dir= output_dir + "frames/";
  boost::filesystem::path frames_dir_path(frames_dir);
  boost::filesystem::create_directory(frames_dir_path);
  std::unordered_set<std::string> fields;
  auto writer = std::make_shared<FrameWriter>(fields, frames_dir, FrameWriter::FileFormat::BINARY);
  writer->SetSource(camera->GetStream());
  writer->SetBlockOnPush(block);
  procs.push_back(writer);

  // Create Throttler.
  auto throttler = std::make_shared<Throttler>(input_fps);
  throttler->SetSource(camera->GetStream());
  procs.push_back(throttler);

  // Create FlowControlEntrance.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(tokens);
  fc_entrance->SetSource(throttler->GetSink());
  procs.push_back(fc_entrance);

  // Create ImageTransformer.
  auto model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(),
                    model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, true);
  transformer->SetSource(fc_entrance->GetSink());
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create NeuralNetEvaluator.
  std::vector<std::string> output_layer_names = {layer};
  auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape, nne_batch_size,
                                                  output_layer_names);
  nne->SetSource(transformer->GetSink());
  nne->SetBlockOnPush(block);
  procs.push_back(nne);
  auto nne_stream = nne->GetSink(layer);

  // Create ImageMatch level 0. Use the same batch size as the NeuralNetEvaluator.
  auto im_0 = std::make_shared<ImageMatch>("", false, nne_batch_size);
  im_0->SetSource(nne_stream);
  im_0->SetBlockOnPush(block);
  im_0->SetQueryMatrix(1, 1, 1024);
  procs.push_back(im_0);

  // FlowControlExit
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(im_0->GetSink());
  procs.push_back(fc_exit);

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  auto reader = fc_exit->GetSink()->Subscribe();
  while(true) {
    auto frame = reader->PopFrame();
    LOG(INFO) << "Frame " << frame->GetValue<unsigned long>("frame_id") << " exiting pipeline";
    if(frame->IsStopFrame()) {
      break;
    }
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("FilterForward");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's configuration files.");
  desc.add_options()("ff-conf,f", po::value<std::string>(),
                     "The file containing the keyframe detector's configuration.");
  desc.add_options()("block,b",
                     "Processors should block when pushing frames.");
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
  std::string camera = args["camera"].as<std::string>();
  int input_fps = args["input-fps"].as<int>();
  unsigned int tokens = args["tokens"].as<unsigned int>();
  std::string layer = args["layer"].as<std::string>();
  std::string model = args["model"].as<std::string>();
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  std::string output_dir = args["output-dir"].as<std::string>();
  Run(ff_conf, block, camera, input_fps, tokens, model, layer, nne_batch_size, output_dir);
  return 0;
}
