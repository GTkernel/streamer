//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#ifndef STREAMER_IP_CAMERA_H
#define STREAMER_IP_CAMERA_H

#include "camera.h"
#include "video/gst_video_capture.h"

class GSTCamera : public Camera {
 public:
  GSTCamera(const string &name, const string &video_uri, int width = -1,
            int height = -1);
  virtual CameraType GetCameraType() const override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  GstVideoCapture capture_;
};

#endif  // STREAMER_IP_CAMERA_H
