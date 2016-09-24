//
// Created by xianran on 9/23/16.
//

#include "camera.h"

Camera::Camera(const string &name, const string &video_uri)
    : name_(name), video_uri_(video_uri) {}

string Camera::GetName() const {
  return name_;
}
string Camera::GetVideoURI() const {
  return video_uri_;
}
