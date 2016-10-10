/**
 * @brief runner.cpp - The long running process on the device. This process
 * manages the cameras and streams, run DNN on realtime camera frames, push
 * stats and video frames to local storage.
 */

#include "camera/camera_manager.h"
#include "common/common.h"
#include "cxxopts/cxxopts.hpp"
#include "linenoise/linenoise.h"
#include "model/model_manager.h"
#include "stream/stream.h"
#include "utils/string_utils.h"

#include <boost/algorithm/string.hpp>
#include <thread>

using std::cout;
using std::endl;

//// Global vars
CameraManager &camera_manager = CameraManager::GetInstance();
ModelManager &model_manager = ModelManager::GetInstance();
std::unordered_map<string, std::shared_ptr<Stream>> streams;

//// Macros
#define CMD_HISTORY_PATH ".cmd_history"

void ListCameras() {
  for (auto &itr : camera_manager.GetCameras()) {
    cout << "Camera: " << itr.second->GetName() << endl
         << "-- URI: " << itr.second->GetVideoURI() << endl;
  }
}

void ListModels() {
  for (auto &itr : model_manager.GetModelDescs()) {
    cout << "Model: " << itr.first << endl
         << "-- desc_path: " << itr.second.GetModelDescPath() << endl
         << "-- param_path: " << itr.second.GetModelParamsPath() << endl;
  }
}

/**
 * @brief Args to imitate command line arguments.
 */
struct Args {
 public:
  Args(const std::vector<string> &tokens) {
    argc = (int)tokens.size();
    argv = new char *[argc];
    for (int i = 0; i < argc; i++) {
      argv[i] = new char[tokens[i].size() + 1];
      strcpy(argv[i], tokens[i].c_str());
    }
  }
  Args(const Args &other) = delete;
  ~Args() {
    for (int i = 0; i < argc; i++) {
      delete argv[i];
    }
    delete[] argv;
  }
  char **argv;
  int argc;
};

static void PreviewStream(const string &name, std::shared_ptr<Stream> stream) {
  cv::namedWindow(name);
  //  while (true) {
  //    cv::Mat frame = stream->PopFrame().GetImage();
  //    cv::imshow(name, frame);
  //    int key = cv::waitKey(10);
  //    if (key == 'q') {
  //      break;
  //    }
  //  }
  cv::destroyWindow(name);
}

void ExecuteCamCommand(const string &subcommand, Args *args) {
  cxxopts::Options options("", "");

  options.parse(args->argc, args->argv);
  // List
  if (subcommand == "list") {
    ListCameras();
  } else {
    LOG(ERROR) << "cam command not recognized";
  }
}

void ExecuteStreamCommand(const string &subcommand, Args *args) {
  cxxopts::Options options("", "");
  options.add_options()("name", "name of the stream", cxxopts::value<string>());

  if (subcommand == "list") {
  } else if (subcommand == "open") {
    LOG(INFO) << "Open stream";
    options.add_options()("camera", "camera name", cxxopts::value<string>());
  } else if (subcommand == "preview") {
    options.parse(args->argc, args->argv);
    string name = options["name"].as<string>();
    auto stream = streams[name];
    PreviewStream(name, stream);
  } else {
    LOG(ERROR) << "stream command not recognized";
  }
}

// void ExecuteEvalCommand(const string &subcommand, Args *args) {
//  cxxopts::Options options("", "");
//  options.add_options()
//      ("name", "evaluator name", cxxopts::value<string>())
//      ("type", "the type of evaluator", cxxopts::value<string>());
//
//  if (subcommand == "create") {
//
//  }
//}

void Execute(const string &line) {
  std::vector<string> tokens;
  boost::split(tokens, line, boost::is_any_of(" \t"), boost::token_compress_on);

  string command = tokens[0];
  string subcommand = tokens[1];
  Args args(tokens);
  if (command == "cam") {
    ExecuteCamCommand(subcommand, &args);
  } else if (command == "stream") {
    ExecuteStreamCommand(subcommand, &args);
    //  } else if (command == "eval") {
    //    ExecuteEvalCommand(subcommand, &args);
  } else {
    LOG(ERROR) << "Command " << command << " is not recognized";
  }
}

int main(int argc, char *argv[]) {
  // Set up evironment
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;
  // Set up linenoise to read command
  char *charline;
  linenoiseSetMultiLine(1);
  if (std::fstream(CMD_HISTORY_PATH)) {
    linenoiseHistoryLoad(CMD_HISTORY_PATH);
  }
  //// Main loop
  while ((charline = linenoise("camnet> ")) != NULL) {
    string line(charline);
    line = TrimSpaces(line);
    linenoiseHistoryAdd(charline);
    linenoiseHistorySave(CMD_HISTORY_PATH);
    Execute(line);

    free(charline);
  }

  return 0;
}