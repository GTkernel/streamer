//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#ifndef STREAMER_VIMBA_CAMERA_H
#define STREAMER_VIMBA_CAMERA_H

#include "camera.h"

#include <VimbaCPP/Include/VimbaCPP.h>

namespace VmbAPI = AVT::VmbAPI;

class VimbaCameraFrameObserver;

/**
 * @brief A class for AlliedVision camera, the name Vimba comes from
 * AlliedVision's Vimba SDK.
 */
class VimbaCamera : Camera {
  friend class VimbaCameraFrameObserver;
 public:
  VimbaCamera(const string &name, const string &video_uri, int width,
              int height);
  virtual CameraType GetCameraType() const override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  VmbAPI::VimbaSystem &vimba_system;
  VmbAPI::CameraPtr camera_;
};

#endif  // STREAMER_VIMBA_CAMERA_H
