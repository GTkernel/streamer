//
// Created by xianran on 9/23/16.
//

#ifndef TX1DNN_CAMERA_MANAGER_H
#define TX1DNN_CAMERA_MANAGER_H

#include "common/common.h"
#include "camera/camera.h"

/**
 * @brief The class that manages and controls all cameras on the device.
 */
class CameraManager {
 public:
  static CameraManager &GetInstance();
 public:
  CameraManager();
  std::vector<std::shared_ptr<Camera>> ListCameras();
 private:
  std::vector<std::shared_ptr<Camera>> cameras_;
};

#endif //TX1DNN_CAMERA_MANAGER_H
