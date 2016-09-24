#include <camera/camera_manager.h>
/**
 * @brief runner.cpp - The long running process on the device. This process
 * manages the cameras and streams, run DNN on realtime camera frames, push
 * stats and video frames to local storage.
 */
int main(int argc, char *argv[]) {
  CameraManager &camera_manager = CameraManager::GetInstance();
  for (auto &camera : camera_manager.ListCameras()) {
    LOG(INFO) << "Camera: " << camera->GetName() << " URI: " << camera->GetVideoURI();
  }
  return 0;
}