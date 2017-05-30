//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera_manager.h"
#include "common/context.h"

// The path to the camera config file
static const string CAMERA_TOML_FILENAME = "cameras.toml";

CameraManager &CameraManager::GetInstance() {
  static CameraManager manager;
  return manager;
}

/**
 * @brief Read from the configurations and initialize the list of cameras.
 */
CameraManager::CameraManager() {
  string camera_toml_path =
      Context::GetContext().GetConfigFile(CAMERA_TOML_FILENAME);
  auto root_value = ParseTomlFromFile(camera_toml_path);

  auto cameras_value = root_value.find("camera")->as<toml::Array>();

  for (const auto &camera_value : cameras_value) {
    CHECK(camera_value.find("name") != nullptr);
    CHECK(camera_value.find("video_uri") != nullptr);

    string name = camera_value.get<string>("name");
    string video_uri = camera_value.get<string>("video_uri");
    string tile_up_command;
    string tile_down_command;
    string pan_left_command;
    string pan_right_command;

    int width = -1;
    int height = -1;
    if (camera_value.find("width") != nullptr) {
      width = camera_value.get<int>("width");
    }
    if (camera_value.find("height") != nullptr) {
      height = camera_value.get<int>("height");
    }
    if (camera_value.find("tile_up_command") != nullptr) {
      tile_up_command = camera_value.get<string>("tile_up_command");
    }
    if (camera_value.find("tile_down_command") != nullptr) {
      tile_down_command = camera_value.get<string>("tile_down_command");
    }
    if (camera_value.find("pan_left_command") != nullptr) {
      pan_left_command = camera_value.get<string>("pan_left_command");
    }
    if (camera_value.find("pan_right_command") != nullptr) {
      pan_right_command = camera_value.get<string>("pan_right_command");
    }

    std::shared_ptr<Camera> camera;
    string video_protocol = SplitString(video_uri, ":")[0];
    if (video_protocol == "gst" || video_protocol == "rtsp" ||
        video_protocol == "file") {
      camera.reset(new GSTCamera(name, video_uri, width, height));
    } else if (video_protocol == "pgr") {
#ifdef USE_PTGRAY
      camera.reset(new PGRCamera(name, video_uri, width, height));
#else
      LOG(WARNING) << "Not built with PtGray FlyCapture SDK, camera: " << name
                   << " is not loaded";
      continue;
#endif
    } else if (video_protocol == "vmb") {
#ifdef USE_VIMBA
      camera.reset(new VimbaCamera(name, video_uri, width, height));
#else
      LOG(WARNING) << "Not built with AlliedVision Vimba SDK, camera: " << name
                   << " is not loaded";
      continue;
#endif
    } else {
      LOG(WARNING) << "Unknown video protocol: " << video_protocol
                   << ". Ignored";
      continue;
    }

    camera->tile_down_command_ = tile_down_command;
    camera->tile_up_command_ = tile_up_command;
    camera->pan_left_command_ = pan_left_command;
    camera->pan_right_command_ = pan_right_command;

    cameras_.emplace(name, camera);
  }
}

std::unordered_map<string, std::shared_ptr<Camera>>
CameraManager::GetCameras() {
  return cameras_;
}

std::shared_ptr<Camera> CameraManager::GetCamera(const string &name) {
  auto itr = cameras_.find(name);
  CHECK(itr != cameras_.end())
      << "Camera with name " << name << " is not present";
  return itr->second;
}

bool CameraManager::HasCamera(const string &name) const {
  return cameras_.count(name) != 0;
}
