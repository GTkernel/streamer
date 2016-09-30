//
// Created by xianran on 9/23/16.
//

#include "camera.h"

Camera::Camera(const string &name, const string &video_uri)
    : name_(name), video_uri_(video_uri), opened_(false) {}

string Camera::GetName() const {
  return name_;
}

string Camera::GetVideoURI() const {
  return video_uri_;
}

bool Camera::Open() {
  if (!opened_) {
    opened_ = capture_.CreatePipeline(video_uri_);
    return opened_;
  }

  return true;
}
void Camera::Close() {
  if (opened_) {
    capture_.DestroyPipeline();
  }
}
cv::Mat Camera::Capture() {
  CHECK(opened_) << "Camera is not open yet";
  return capture_.GetFrame();
}
