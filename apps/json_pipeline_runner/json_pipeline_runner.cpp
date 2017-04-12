
#include <boost/program_options.hpp>

#include "pipeline/pipeline.h"
#include "streamer.h"



void Run(const string pipeline_filepath, bool display) {
  std::ifstream i(pipeline_filepath);
  nlohmann::json json;
  i >> json;

  std::shared_ptr<Pipeline> pipeline = Pipeline::ConstructPipeline(json);

  printf("Before Pipeline::Start()\n");
  pipeline->Start();
  printf("After Pipeline::Start()\n");

  while(true) {};
}

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  boost::program_options::options_description desc(
      "Runs the pipeline described by a JSON file");
  desc.add_options()("help,h", "Print the help message");
  desc.add_options()("pipeline-file,f",
                     boost::program_options::value<string>()->value_name(
		         "PIPELINE-FILE")->required(),
                     "Path to the JSON file describing a pipeline");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()("device", boost::program_options::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()("config-dir,C",
                     boost::program_options::value<string>()->value_name("CONFIG-DIR"),
		     "The directory to find streamer's configurations");

  boost::program_options::variables_map args;
  try {
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), args);
    boost::program_options::notify(args);
  } catch (const boost::program_options::error &e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (args.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<string>());
  }

  int device_number = args["device"].as<int>();
  string pipeline_filepath = args["pipeline-file"].as<string>();
  bool display = args.count("display") != 0;

  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);

  Run(pipeline_filepath, display);
  return 0;
}
