//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#ifndef STREAMER_CAMERA_CAMERA_MANAGER_H_
#define STREAMER_CAMERA_CAMERA_MANAGER_H_

#include <unordered_map>
#include "camera/camera.h"
#include "common/common.h"
#include "gst_camera.h"

#ifdef USE_PTGRAY
#include "pgr_camera.h"
#endif
#ifdef USE_VIMBA
#include "vimba_camera.h"
#endif

/**
 * @brief The class that manages and controls all cameras on the device.
 */
class CameraManager {
 public:
  static CameraManager& GetInstance();

 public:
  CameraManager();
  CameraManager(const CameraManager& other) = delete;
  std::unordered_map<string, std::shared_ptr<Camera>> GetCameras();
  std::shared_ptr<Camera> GetCamera(const string& name);
  bool HasCamera(const string& name) const;

 private:
  std::unordered_map<string, std::shared_ptr<Camera>> cameras_;
};

#endif  // STREAMER_CAMERA_CAMERA_MANAGER_H_
