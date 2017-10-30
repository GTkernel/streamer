// Deploy a pipeline from a JSON specification

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>
#include <json/src/json.hpp>

#include "pipeline/pipeline.h"

namespace po = boost::program_options;

void Run(const std::string& pipeline_filepath) {
  std::ifstream i(pipeline_filepath);
  nlohmann::json json;
  i >> json;

  std::shared_ptr<Pipeline> pipeline = Pipeline::ConstructPipeline(json);
  pipeline->Start();

  std::cout << "Press \"Enter\" to stop." << std::endl;
  getchar();

  pipeline->Stop();
}

int main(int argc, char* argv[]) {
  po::options_description desc("Runs a pipeline described by a JSON file");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("pipeline,p", po::value<std::string>()->required(),
                     "Path to a JSON file describing a pipeline.");

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

  std::string pipeline_filepath = args["pipeline"].as<std::string>();
  Run(pipeline_filepath);
  return 0;
}
