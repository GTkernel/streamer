//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera_manager.h"
#include "common/context.h"
#include "gst_camera.h"

#ifdef USE_PTGRAY
#include "pgr_camera.h"
#endif

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

    int width = -1;
    int height = -1;
    if (camera_value.find("width") != nullptr) {
      width = camera_value.get<int>("width");
    }
    if (camera_value.find("height") != nullptr) {
      height = camera_value.get<int>("height");
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
      LOG(FATAL) << "Not built with PtGray FlyCapture SDK";
#endif
    } else {
      LOG(FATAL) << "Unknown video protocol: " << video_protocol;
    }

    cameras_.emplace(name, camera);
  }
}

std::unordered_map<string, std::shared_ptr<Camera>>
CameraManager::GetCameras() {
  return cameras_;
}

std::shared_ptr<Camera> CameraManager::GetCamera(const string &name) {
  auto itr = cameras_.find(name);
  CHECK(itr != cameras_.end()) << "Camera with name " << name
                               << " is not present";
  return itr->second;
}

bool CameraManager::HasCamera(const string &name) const {
  return cameras_.count(name) != 0;
}
