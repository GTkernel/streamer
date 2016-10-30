//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include "gst_camera.h"

GSTCamera::GSTCamera(const string &name, const string &video_uri, int width,
                     int height)
    : Camera(name, video_uri, width, height) {}

bool GSTCamera::Init() {
  bool opened = capture_.CreatePipeline(video_uri_);

  if (!opened) {
    LOG(INFO) << "can't open camera";
    return false;
  }

  return true;
}
bool GSTCamera::OnStop() {
  capture_.DestroyPipeline();
  return true;
}
void GSTCamera::Process() {
  cv::Mat frame = capture_.GetFrame();
  PushFrame(0, new ImageFrame(frame, frame));
}
CameraType GSTCamera::GetType() const { return CAMERA_TYPE_GST; }
