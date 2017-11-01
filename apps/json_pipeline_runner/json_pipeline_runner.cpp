// Deploy a pipeline from a JSON specification

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>
#ifdef USE_GRAPHVIZ
#include <graphviz/gvc.h>
#endif
#include <json/src/json.hpp>

#include "pipeline/pipeline.h"

namespace po = boost::program_options;

#ifdef USE_GRAPHVIZ
static std::atomic<bool> done(false);

static std::shared_ptr<std::thread> ShowGraph(
    std::shared_ptr<Pipeline> pipeline) {
  std::string graph = pipeline->GetGraph();
  std::cout << "Pipeline:\n" + graph << std::endl;

  GVC_t* gvc = gvContext();
  Agraph_t* dg = agmemread(graph.c_str());
  CHECK(dg != NULL);

  int err = gvLayout(gvc, dg, "dot");
  CHECK(err == 0);

  char* buf;
  unsigned int len;
  err = gvRenderData(gvc, dg, "bmp", &buf, &len);
  CHECK(err == 0);
  buf = (char*)realloc(buf, len + 1);

  std::vector<char> data(buf, buf + len);
  gvFreeRenderData(buf);

  cv::Mat data_mat(data, true);
  cv::Mat img = cv::imdecode(data_mat, cv::IMREAD_COLOR);

  auto t = std::make_shared<std::thread>([img] {
    while (!done) {
      cv::imshow("Graph", img);
      cv::waitKey(10);
    }
  });

  return t;
}
#endif

void Run(const std::string& pipeline_filepath, bool dry_run, bool show_graph) {
  std::ifstream i(pipeline_filepath);
  nlohmann::json json;
  i >> json;

  std::cout << "Pipeline:\n" + json.dump(4) << std::endl;

  std::shared_ptr<Pipeline> pipeline = Pipeline::ConstructPipeline(json);

#ifdef USE_GRAPHVIZ
  auto t = show_graph ? ShowGraph(pipeline) : nullptr;
#endif

  if (!dry_run) {
    pipeline->Start();
  }

  if (!dry_run || show_graph) {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
  }

#ifdef USE_GRAPHVIZ
  if (show_graph) {
    done = true;
    t->join();
  }
#endif

  if (!dry_run) {
    pipeline->Stop();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("Runs a pipeline described by a JSON file");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing Streamer's config files.");
  desc.add_options()("pipeline,p", po::value<std::string>()->required(),
                     "Path to a JSON file describing a pipeline.");
  desc.add_options()("dry-run,n", "Create pipeline but do not run it");
#ifdef USE_GRAPHVIZ
  desc.add_options()("graph,g", "Display pipeline graph visually");
#endif

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
  bool dry_run = args.count("dry-run");
  bool show_graph = false;
#ifdef USE_GRAPHVIZ
  show_graph = args.count("graph");
#endif
  std::cout << show_graph << std::endl;
  Run(pipeline_filepath, dry_run, show_graph);
  return 0;
}
