/**
 * @brief json_pipeline_runner.cpp - Deploy pipeline from JSON spec
 */

#include <cstdio>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "pipeline/pipeline.h"

namespace po = boost::program_options;

/**
 * @brief Deploy pipeline from JSON Spec
 *
 */
void Run(std::string pipeline_filepath) {
  std::ifstream i(pipeline_filepath);
  nlohmann::json json;
  i >> json;

  std::shared_ptr<Pipeline> pipeline = Pipeline::ConstructPipeline(json);
  pipeline->Start();

  while(true) {};
}

int main(int argc, char* argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Runs the pipeline described by a JSON file");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config_dir,C", po::value<string>(),
                     "The directory to find streamer's configuration");
  desc.add_options()("pipeline_file,f", po::value<string>(),
                     "Path to the JSON file describing a pipeline");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  //// Parse arguments

  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto pipeline_filepath = vm["pipeline_file"].as<string>();

  Run(pipeline_filepath);

  return 0;
}

