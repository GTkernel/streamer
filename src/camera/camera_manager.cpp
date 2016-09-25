//
// Created by xianran on 9/23/16.
//

#include "camera_manager.h"

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
  string camera_toml_path = Context::GetContext().GetConfigFile(CAMERA_TOML_FILENAME);
  auto root_value = ParseTomlFromFile(camera_toml_path);

  auto cameras_value = root_value.find("camera")->as<toml::Array>();

  for (const auto &camera_value : cameras_value) {
    string name = camera_value.get<string>("name");
    string video_uri = camera_value.get<string>("video_uri");
    std::shared_ptr<Camera> camera(new Camera(name, video_uri));
    LOG(INFO) << "Camera - name: " << name << " " << "uri: " << video_uri;
    cameras_.emplace(name, camera);
  }
}

std::unordered_map<string,
                   std::shared_ptr<Camera>> CameraManager::GetCameras() {
  return cameras_;
}

std::shared_ptr<Camera> CameraManager::GetCamera(const string &name) {
  auto itr = cameras_.find(name);
  CHECK(itr != cameras_.end()) << "Camera with name " << name
                               << " is not present";
  return itr->second;
}
