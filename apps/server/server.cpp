/**
 * @brief runner.cpp - The long running process on the device. This process
 * manages the cameras and streams, run DNN on realtime camera frames, push
 * stats and video frames to local storage.
 */

#include <boost/program_options.hpp>
#include <csignal>

#include "server_utils.h"
#include "streamer.h"

namespace po = boost::program_options;

#define STRING_PATTERN "([a-zA-Z0-9_]+)"

static void SetUpEndpoints(HttpServer &server) {
  // GET /hello
  server.resource["^/hello$"]["GET"] = [](HttpServerResponse response,
                                          HttpServerRequest request) {
    string content = "Hello from streamer";
    LOG(INFO) << "Here";
    Send200Response(response, content);
  };

  // GET /cameras
  server.resource["^/cameras"]["GET"] = [](HttpServerResponse response,
                                           HttpServerRequest request) {
    CameraManager &camera_manager = CameraManager::GetInstance();
    ModelManager &model_manager = ModelManager::GetInstance();
    std::vector<pt::ptree> cameras_node;
    auto cameras = camera_manager.GetCameras();
    for (auto itr = cameras.begin(); itr != cameras.end(); itr++) {
      pt::ptree node;
      CameraToJson(itr->second.get(), node);
      cameras_node.push_back(node);
    }

    Send200Response(response, ListToJson("cameras", cameras_node));
  };

  // POST /cameras/:cam_name/configure
  server.resource["^/cameras/" STRING_PATTERN "/configure$"]["POST"] = [](
      HttpServerResponse response, HttpServerRequest request) {
    string camera_name = request->path_match[1];
    auto camera = CameraManager::GetInstance().GetCamera(camera_name);

    pt::ptree doc;
    pt::read_json(request->content, doc);

#ifdef USE_PTGRAY
    if (camera->GetType() != CAMERA_TYPE_PTGRAY) {
      LOG(WARNING) << "Non-PtGray camera control not implemented";
      Send400Response(response, "Not implemented");
      return;
    }

    auto ptgray_camera = std::dynamic_pointer_cast<PGRCamera>(camera);
    if (doc.count("width") && doc.count("height") && doc.count("video_mode")) {
      Shape shape;
      FlyCapture2::Mode mode = FlyCapture2::MODE_0;
      shape.width = doc.get<int>("width");
      shape.height = doc.get<int>("height");

      string mode_str = doc.get<string>("video_mode");
      if (mode_str == "mode_0") {
        mode = FlyCapture2::MODE_0;
      } else if (mode_str == "mode_1") {
        mode = FlyCapture2::MODE_1;
      } else {
        string warning_message = mode_str + "is not a supported mode";
        LOG(WARNING) << warning_message;
        return;
      }

      ptgray_camera->SetImageSizeAndVideoMode(shape, mode);
    }

    if (doc.count("sharpness")) {
      float sharpness = doc.get<float>("sharpness");
      ptgray_camera->SetSharpness(sharpness);
    }

    if (doc.count("exposure")) {
      float exposure = doc.get<float>("exposure");
      ptgray_camera->SetExposure(exposure);
    }
#else
    LOG(WARNING) << "Not built with PtGray, not able to control camera";
    Send400Response(response, "Not built with PtGray");
#endif
  };

  // GET /cameras/:cam_name/stream
  server.resource["^/cameras/" STRING_PATTERN "/stream$"]["GET"] = [](
      HttpServerResponse response, HttpServerRequest request) {

  };

  // GET /pipelines => List all pipelines

  // POST /pipelines/run => Run a new pipeline, this include vision features,
  // DNN evaluation, video recording, video streaming, etc.

  // DELETE /pipelines/stop => Kill an existing pipeline

  // Get /pipelines/:pipeline_name/stream => Stream a pipeline                                                              
}

int main(int argc, char *argv[]) {
  // Set up gstreamer and google-log
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Streamer server that exports an HTTP API");
  desc.add_options()(
      "config_dir,C",
      po::value<string>()->value_name("CONFIG_DIR")->default_value("./config"),
      "The directory to find streamer's configurations");
  desc.add_options()("port,p",
                     po::value<int>()->value_name("PORT")->default_value(15213),
                     "The port to bind streamer serveer");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  Context::GetContext().Init();

  size_t server_thread_num = 1;
  unsigned short server_port = (unsigned short)vm["port"].as<int>();
  HttpServer server(server_port, server_thread_num);

  SetUpEndpoints(server);

  // Start the server thread
  std::thread server_thread([&server]() { server.start(); });

  STREAMER_SLEEP(1000);
  LOG(INFO) << "Streamer server started at " << server_port;
  server_thread.join();
}