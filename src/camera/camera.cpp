//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera.h"

Camera::Camera(const string &name, const string &video_uri, int width,
               int height, int nsink)
    : Processor({}, nsink),
      name_(name),
      video_uri_(video_uri),
      width_(width),
      height_(height) {
  stream_ = sinks_[0];
}

string Camera::GetName() const { return name_; }

string Camera::GetVideoURI() const { return video_uri_; }

int Camera::GetWidth() { return width_; }
int Camera::GetHeight() { return height_; }

std::shared_ptr<Stream> Camera::GetStream() const { return stream_; }

ProcessorType Camera::GetType() {
  return PROCESSOR_TYPE_CAMERA;
}
