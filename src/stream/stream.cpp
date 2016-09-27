//
// Created by xianran on 9/26/16.
//

#include "stream.h"

Stream::Stream(const std::shared_ptr<Camera> camera) : camera_(camera) {
  capture_.CreatePipeline(camera->GetVideoURI());
}

cv::Mat Stream::GetFrame() {
  return capture_.GetFrame();
}
