#include "camera/camera_manager.h"
#include "model/model_manager.h"
/**
 * @brief runner.cpp - The long running process on the device. This process
 * manages the cameras and streams, run DNN on realtime camera frames, push
 * stats and video frames to local storage.
 */

int main(int argc, char *argv[]) {
  // Get camera manager
  CameraManager &camera_manager = CameraManager::GetInstance();
  for (auto &itr: camera_manager.GetCameras()) {
    LOG(INFO) << "Camera: " << itr.second->GetName() << " URI: "
              << itr.second->GetVideoURI();
  }
  // Get model manager
  ModelManager &model_manager = ModelManager::GetInstance();
  for (auto &itr : model_manager.GetModelDescs()) {
    LOG(INFO) << "Model: " << itr.first;
    LOG(INFO) << "-- desc_path: " << itr.second.GetModelDescPath();
    LOG(INFO) << "-- param_path: " << itr.second.GetModelParamsPath();
  }
  return 0;
}