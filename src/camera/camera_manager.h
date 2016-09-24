//
// Created by xianran on 9/23/16.
//

#ifndef TX1DNN_CAMERA_MANAGER_H
#define TX1DNN_CAMERA_MANAGER_H

#include "common/common.h"
#include "camera/camera.h"
#include <unordered_map>

/**
 * @brief The class that manages and controls all cameras on the device.
 */
class CameraManager {
 public:
  static CameraManager &GetInstance();
 public:
  CameraManager();
  std::unordered_map<string, std::shared_ptr<Camera>> GetCameras();
  std::shared_ptr<Camera> GetCamera(const string &name);
 private:
  std::unordered_map<string, std::shared_ptr<Camera>> cameras_;
};

#endif //TX1DNN_CAMERA_MANAGER_H
