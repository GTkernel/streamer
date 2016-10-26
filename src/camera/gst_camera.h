//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#ifndef TX1DNN_IP_CAMERA_H
#define TX1DNN_IP_CAMERA_H

#include "camera.h"
#include "video/gst_video_capture.h"

class GSTCamera : public Camera {
 public:
  GSTCamera(const string &name, const string &video_uri, int width = -1,
            int height = -1);

 protected:
  virtual bool Init();
  virtual bool OnStop();
  virtual void Process();

 private:
  GstVideoCapture capture_;
};

#endif  // TX1DNN_IP_CAMERA_H
