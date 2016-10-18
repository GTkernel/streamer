//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera.h"

Camera::Camera(const string &name, const string &video_uri, int width,
               int height)
    : name_(name),
      video_uri_(video_uri),
      width_(width),
      height_(height),
      stream_(new Stream(name)) {
  sinks_.push_back(stream_);
}

string Camera::GetName() const { return name_; }

string Camera::GetVideoURI() const { return video_uri_; }

bool Camera::Init() {
  bool opened = capture_.CreatePipeline(video_uri_);

  if (!opened) {
    LOG(INFO) << "can't open camera";
    return false;
  }

  return true;
}

bool Camera::OnStop() {
  capture_.DestroyPipeline();
  return true;
}

void Camera::Process() {
  cv::Mat frame = capture_.GetFrame();
  PushFrame(0, new ImageFrame(frame, frame));
}

int Camera::GetWidth() { return width_; }
int Camera::GetHeight() { return height_; }

std::shared_ptr<Stream> Camera::GetStream() const { return stream_; }
