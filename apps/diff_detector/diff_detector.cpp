
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/types.h"
#include "processor/diff_detector.h"
#include "processor/image_transformer.h"
#include "processor/processor.h"
#include "utils/file_utils.h"

namespace po = boost::program_options;

void Run(const std::string& camera_name, int dim, double threshold,
         bool blocked, int block_size, const std::string& weights_path,
         bool dynamic_ref, long t_diff_frames, const std::string& ref_path,
         const std::string& output_dir) {
  std::vector<std::shared_ptr<Processor>> procs;

  // Create Camera.
  std::shared_ptr<Camera> camera =
      CameraManager::GetInstance().GetCamera(camera_name);
  procs.push_back(camera);

  // Create ImageTransformer.
  auto transformer =
      std::make_shared<ImageTransformer>(Shape(3, dim, dim), true, true);
  transformer->SetSource(camera->GetStream());
  procs.push_back(transformer);

  // Create DiffDetector.
  std::shared_ptr<DiffDetector> diffd;
  if (blocked) {
    if (dynamic_ref) {
      diffd = std::make_shared<DiffDetector>(threshold, block_size,
                                             weights_path, t_diff_frames);
    } else {
      diffd = std::make_shared<DiffDetector>(threshold, block_size,
                                             weights_path, ref_path);
    }
  } else {
    if (dynamic_ref) {
      diffd = std::make_shared<DiffDetector>(threshold, t_diff_frames);
    } else {
      diffd = std::make_shared<DiffDetector>(threshold, ref_path);
    }
  }
  diffd->SetSource(transformer->GetSink());
  procs.push_back(diffd);

  // Subscribe before starting the processors so that we definitely do not miss
  // any frames.
  StreamReader* reader = diffd->GetSink()->Subscribe();

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start();
  }

  std::ofstream micros_log(output_dir + "/diff_micros.txt");
  while (true) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr) {
      if (frame->IsStopFrame()) {
        break;
      }

      if (frame->Count("DiffDetector.diff_micros")) {
        micros_log << frame->GetValue<long>("DiffDetector.diff_micros") << "\n";
      }
    }
  }
  micros_log.close();

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Evaluate difference detectors");
  desc.add_options()("help,h", "print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use.");
  desc.add_options()("dim,x", po::value<int>()->required(),
                     "The square size to which the incoming frames will be "
                     "resized.");
  desc.add_options()("threshold,h", po::value<double>()->required(),
                     "The difference threshold.");
  desc.add_options()("blocked,b", "Whether to use the blocked MSE algorithm.");
  desc.add_options()("block-size,s", po::value<int>(),
                     "Block size, in pixels, to use with \"--blocked\".");
  desc.add_options()("weights,w", po::value<std::string>(),
                     "The path to the weights file is using \"--blocked\".");
  desc.add_options()("dynamic-ref,d", "Use a dynamic reference image.");
  desc.add_options()("t-diff-frames,t", po::value<unsigned long>(),
                     "Time delta (in frames) to use with \"--dynamic-ref\".");
  desc.add_options()("ref,r", po::value<std::string>(),
                     "Path to the static reference image if not using "
                     "\"--dynamic-ref\".");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "THe directory in which to store the results files.");

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
  auto dim = args["dim"].as<int>();
  if (dim < 1) {
    std::ostringstream msg;
    msg << "Value for \"--dim\" must be greater than 0, but is: " << dim;
    throw std::invalid_argument(msg.str());
  }
  auto threshold = args["threshold"].as<double>();
  bool blocked = args.count("blocked");
  int block_size = 0;
  if (blocked) {
    if (!args.count("block-size")) {
      throw std::runtime_error("\"--blocked\" requires \"--block-size\"");
    }
    block_size = args["block-size"].as<int>();
  }
  std::string weights_path;
  if (blocked) {
    if (!args.count("weights")) {
      throw std::runtime_error("\"--blocked\" requires \"--weights\"");
    }
    weights_path = args["weights"].as<std::string>();
  }
  bool dynamic_ref = args.count("dynamic-ref");
  long t_diff_frames = 0;
  std::string ref_path;
  if (dynamic_ref) {
    if (!args.count("t-diff-frames")) {
      throw std::runtime_error(
          "\"--dynamic-ref\" requires \"--t-diff-frames\"");
    }
    t_diff_frames = args["t-diff-frames"].as<unsigned long>();
  } else {
    ref_path = args["ref"].as<std::string>();
  }
  auto output_dir = args["output-dir"].as<std::string>();
  CHECK(DirExists(output_dir))
      << "\"" << output_dir << "\" is not a directory!";

  Run(camera_name, dim, threshold, blocked, block_size, weights_path,
      dynamic_ref, t_diff_frames, ref_path, output_dir);
}
