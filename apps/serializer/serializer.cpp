/**
 * @brief serializer.cpp - An example application to serialize Frames to disk
 */

#include <csignal>

#include <boost/program_options.hpp>

#include "camera/camera_manager.h"
#include "common/context.h"
#include "processor/file_writer.h"

namespace po = boost::program_options;

std::shared_ptr<Camera> camera;
std::shared_ptr<FileWriter> fw;

void SignalHandler(int) {
  std::cout << "Received SIGINT, stopping" << std::endl;
  if (camera != nullptr) camera->Stop();
  if (fw != nullptr) fw->Stop();

  exit(0);
}

void Run(const string& camera_name, const string& file_name,
         const FileWriter::file_format& format) {
  CameraManager& camera_manager = CameraManager::GetInstance();

  camera = camera_manager.GetCamera(camera_name);

  fw = std::make_shared<FileWriter>(file_name, format);
  fw->SetSource("input", camera->GetStream());

  fw->Start();
  camera->Start();

  LOG(INFO) << "Writing frames to `" << file_name << "`";

  while (true) {
  }

  camera->Stop();
  fw->Stop();
}

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc_visible("Simple camera to file serializer");
  desc_visible.add_options()("help,h", "print the help message");
  desc_visible.add_options()("camera,c", po::value<string>()->required(),
                             "The name of the camera to use");
  desc_visible.add_options()("format,f",
                             po::value<string>()->default_value("text"),
                             "The output file format ('text' or 'binary')");
  desc_visible.add_options()("config_dir,C", po::value<string>(),
                             "The directory to find streamer's configurations");

  po::options_description desc_hidden("Hidden options");
  desc_hidden.add_options()("file", po::value<string>()->required(),
                            "The name of the output file");

  po::options_description cmdline_options;
  cmdline_options.add(desc_visible).add(desc_hidden);

  po::options_description visible_options;
  visible_options.add(desc_visible);

  po::positional_options_description pos;
  pos.add("file", 1);

  po::variables_map vm;
  try {
    auto parsed = po::command_line_parser(argc, argv)
                      .options(cmdline_options)
                      .positional(pos)
                      .run();

    for (auto const& opt : parsed.options) {
      if ((opt.position_key == -1) && (opt.string_key == "file")) {
        throw po::unknown_option("file");
      }
    }

    po::store(parsed, vm);

    if (vm.count("help")) {
      std::cerr << "Usage: serializer [options] file_name" << std::endl;
      std::cerr << visible_options << std::endl;
      return 1;
    }

    if (!vm.count("file")) {
      std::cerr << "Missing file_name" << std::endl;
      std::cerr << "Usage: serializer [options] file_name" << std::endl;
      std::cerr << visible_options << std::endl;
      return 1;
    }

    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Usage: serializer [options] file_name" << std::endl;
    std::cerr << visible_options << std::endl;
    return 1;
  }

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<string>();
  auto file_name = vm["file"].as<string>();
  auto file_format = vm["format"].as<string>();

  FileWriter::file_format format;
  if (file_format == "text") {
    format = FileWriter::file_format::TEXT;
  } else if (file_format == "binary") {
    format = FileWriter::file_format::BINARY;
  } else {
    LOG(FATAL) << "Invalid output file format: " << file_format;
  }

  std::signal(SIGINT, SignalHandler);

  Run(camera_name, file_name, format);

  return 0;
}
