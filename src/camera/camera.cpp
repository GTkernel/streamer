//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera.h"

Camera::Camera(const string &name, const string &video_uri)
    : name_(name), video_uri_(video_uri), opened_(false), stream_(new Stream) {}

string Camera::GetName() const { return name_; }

string Camera::GetVideoURI() const { return video_uri_; }

bool Camera::Start() {
  if (opened_) return true;
  opened_ = capture_.CreatePipeline(video_uri_);
  if (!opened_) {
    return false;
  }

  capture_thread_.reset(new std::thread(&Camera::CaptureLoop, this));
  return true;
}
bool Camera::Stop() {
  opened_ = false;
  capture_thread_->join();

  return true;
}

void Camera::CaptureLoop() {
  CHECK(opened_) << "Camera is not open yet";
  while (opened_) {
    cv::Mat frame = capture_.GetFrame();
    stream_->PushFrame(std::shared_ptr<ImageFrame>(new ImageFrame(frame, frame)));
  }
}

std::shared_ptr<Stream> Camera::GetStream() const { return stream_; }
