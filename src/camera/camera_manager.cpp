//
// Created by xianran on 9/23/16.
//

#include "camera_manager.h"

// The path to the camera config file
static const string CAMERA_TOML_PATH = "config/cameras.toml";

CameraManager &CameraManager::GetInstance() {
  static CameraManager manager;
  return manager;
}

/**
 * @brief Read from the configurations and initialize the list of cameras.
 */
CameraManager::CameraManager() {
  auto root_value = ParseTomlFromFile(CAMERA_TOML_PATH);

  auto cameras_value = root_value.find("camera")->as<toml::Array>();

  for (const auto &camera_value : cameras_value) {
    string name = camera_value.get<string>("name");
    string uri = camera_value.get<string>("video_uri");
    std::shared_ptr<Camera> camera(new Camera(name, uri));
    LOG(INFO) << "Camera - name: " << name << " " << "uri: " << uri;
    cameras_.push_back(camera);
  }
}

std::vector<std::shared_ptr<Camera>> CameraManager::ListCameras() {
  return cameras_;
}
