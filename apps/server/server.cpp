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

std::unordered_map<string, StreamPtr> pipelines;

static void SetUpEndpoints(HttpServer &server) {
  auto &camera_manager = CameraManager::GetInstance();

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

    if (camera->GetCameraType() != CAMERA_TYPE_PTGRAY) {
      LOG(WARNING) << "Non-PtGray camera control not implemented";
      Send400Response(response, "Not implemented");
      return;
    }

    if (doc.count("width") && doc.count("height") && doc.count("video_mode")) {
      Shape shape;
      CameraModeType mode;
      shape.width = doc.get<int>("width");
      shape.height = doc.get<int>("height");

      string mode_str = doc.get<string>("video_mode");
      if (mode_str == "mode_0") {
        mode = CAMERA_MODE_0;
      } else if (mode_str == "mode_1") {
        mode = CAMERA_MODE_1;
      } else {
        string warning_message = mode_str + "is not a supported mode";
        LOG(WARNING) << warning_message;
        return;
      }
      camera->SetImageSizeAndMode(shape, mode);
    }

    if (doc.count("sharpness")) {
      float sharpness = doc.get<float>("sharpness");
      camera->SetSharpness(sharpness);
    }

    if (doc.count("exposure")) {
      float exposure = doc.get<float>("exposure");
      camera->SetExposure(exposure);
    }

    Send200Response(response, "success");
  };

  // GET /cameras/:cam_name/capture
  server.resource["^/cameras/" STRING_PATTERN "/capture$"]["GET"] =
      [&camera_manager, &server](HttpServerResponse response,
                                 HttpServerRequest request) {
        string camera_name = request->path_match[1];
        LOG(INFO) << "Received " << request->path;
        if (camera_manager.HasCamera(camera_name)) {
          auto camera = camera_manager.GetCamera(camera_name);
          cv::Mat image;
          if (!camera->Capture(image)) {
            Send400Response(response, "Failed to capture image");
          } else {
            // Compress image to JPEG
            std::vector<uchar> buf;
            cv::imencode(".jpeg", image, buf);
            SendBytes(server, response, (char *)buf.data(), buf.size(),
                      "image/jpeg");
          }
        } else {
          Send400Response(response, "Camera not found: " + camera_name);
        }
      };

  // GET /cameras/:cam_name/ => Get camera information
  // TODO: this is of low priority right now

  // POST /pipelines/run => Run a new pipeline, this include vision features,
  // DNN evaluation, video recording, video streaming, etc.
  // data: spl
  server.resource["^/pipelines/run"]["POST"] = [](HttpServerResponse response, HttpServerRequest request) {
    pt::ptree doc;
    pt::read_json(request->content, doc);

    string pipeline_name = doc.get<string>("name");
    string spl = doc.get<string>("spl");
    SPLParser parser;
    std::vector<SPLStatement> statements;
    bool result = parser.Parse(spl, statements);

    if (!result) {
      Send400Response(response, "Can't parse SPL");
      return;
    }

    auto pipeline = Pipeline::ConstructPipeline(statements);

    if (pipeline == nullptr) {
      Send400Response(response, "Can't construct pipeline");
      return;
    }
  };

  // DELETE /pipelines/stop => Kill an existing pipeline

  // GET /pipelines => List all pipelines

  // Get /pipelines/:pipeline_name/stream => Stream a pipeline
  // data: processor_name, stream_name
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
                     "The port to bind streamer server");

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