//
// Created by xianran on 9/26/16.
//

#ifndef TX1DNN_STREAM_H
#define TX1DNN_STREAM_H

#include "camera/camera.h"
#include "common/common.h"

/**
 * @brief A stream is a serious of data, the data itself could be stats, images,
 * or simply a bunch of bytes.
 */
class Stream {
 public:
  Stream(const std::shared_ptr<Camera> camera);
  cv::Mat GetFrame();
 private:
  const std::shared_ptr<Camera> camera_;
};

#endif //TX1DNN_STREAM_H
