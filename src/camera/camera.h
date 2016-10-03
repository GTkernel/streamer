//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#ifndef TX1DNN_CAMERA_H
#define TX1DNN_CAMERA_H

#include "common/common.h"
#include "stream/stream.h"
#include "video/gst_video_capture.h"

/**
 * @brief This class represents a camera available on the device.
 */
class Camera {
 public:
  Camera(){};
  Camera(const string &name, const string &video_uri);
  string GetName() const;
  string GetVideoURI() const;
  std::shared_ptr<Stream> GetStream() const;
  bool Start();
  bool Stop();

 private:
  void CaptureLoop();
  string name_;
  string video_uri_;
  bool opened_;
  std::unique_ptr<std::thread> capture_thread_;
  GstVideoCapture capture_;
  std::shared_ptr<Stream> stream_;
};

#endif  // TX1DNN_CAMERA_H
