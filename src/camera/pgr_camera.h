//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#ifndef TX1DNN_PGR_CAMERA_H
#define TX1DNN_PGR_CAMERA_H

#include "camera.h"

/**
 * @brief A class for ptgray camera, in order to use this class, you have to
 * make sure that the SDK for ptgray camera is installed on the system.
 */
class PGRCamera : public Camera {
 public:
  PGRCamera(const string &name, const string &video_uri, int width = -1,
            int height = -1);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;
};

#endif  // TX1DNN_PGR_CAMERA_H
