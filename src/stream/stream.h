//
// Created by xianran on 9/26/16.
//

#ifndef TX1DNN_STREAM_H
#define TX1DNN_STREAM_H

#include "camera/camera.h"
#include "common/common.h"
#include "video/gst_video_capture.h"

class Stream {
 public:
  Stream(const std::shared_ptr<Camera> camera);
  cv::Mat GetFrame();
 private:
  const std::shared_ptr<Camera> camera_;
  GstVideoCapture capture_;
};

#endif //TX1DNN_STREAM_H
