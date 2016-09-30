//
// Created by xianran on 9/26/16.
//

#include "stream.h"

Stream::Stream(const std::shared_ptr<Camera> camera) : camera_(camera) {}

cv::Mat Stream::GetFrame() {
  return camera_->Capture();
}
