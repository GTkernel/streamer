//
// Created by xianran on 9/23/16.
//

#ifndef TX1DNN_CAMERA_H
#define TX1DNN_CAMERA_H

#include "common/common.h"

/**
 * @brief This class represents a camera available on the device.
 */
class Camera {
 public:
  Camera() {};
  Camera(const string &id, const string &video_uri);
  string GetName() const;
  string GetVideoURI() const;
 private:
  string name_;
  string video_uri_;
};

#endif //TX1DNN_CAMERA_H
