//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#ifndef STREAMER_VIMBA_CAMERA_H
#define STREAMER_VIMBA_CAMERA_H

#include "camera.h"

#include <VimbaCPP/Include/VimbaCPP.h>
#include <VimbaImageTransform/Include/VmbTransform.h>

#define CHECK_VIMBA(cmd)                                                       \
  do {                                                                         \
    VmbErrorType error;                                                        \
    error = (cmd);                                                             \
    if (error != VmbErrorSuccess) {                                            \
      char error_info[256];                                                    \
      VmbGetErrorInfo(error, (VmbANSIChar_t *)error_info, sizeof(error_info)); \
      LOG(FATAL) << "VIMBA Error happened: " << error << " (" << error_info    \
                 << ")";                                                       \
    }                                                                          \
  } while (0)

namespace VmbAPI = AVT::VmbAPI;

class VimbaCameraFrameObserver;

/**
 * @brief A class for AlliedVision camera, the name Vimba comes from
 * AlliedVision's Vimba SDK.
 */
class VimbaCamera : public Camera {
  friend class VimbaCameraFrameObserver;

 public:
  VimbaCamera(const string &name, const string &video_uri, int width,
              int height, CameraModeType mode = CAMERA_MODE_0,
              CameraPixelFormatType pixel_format = CAMERA_PIXEL_FORMAT_RAW8);
  virtual CameraType GetCameraType() const override;

  virtual float GetExposure() override;
  virtual void SetExposure(float exposure) override;
  virtual float GetSharpness() override;
  virtual void SetSharpness(float sharpness) override;
  virtual Shape GetImageSize() override;
  virtual void SetBrightness(float brightness) override;
  virtual float GetBrightness() override;
  virtual void SetShutterSpeed(float shutter_speed) override;
  virtual float GetShutterSpeed() override;
  virtual void SetSaturation(float saturation) override;
  virtual float GetSaturation() override;
  virtual void SetHue(float hue) override;
  virtual float GetHue() override;
  virtual void SetGain(float gain) override;
  virtual float GetGain() override;
  virtual void SetGamma(float gamma) override;
  virtual float GetGamma() override;
  virtual void SetWBRed(float wb_red) override;
  virtual float GetWBRed() override;
  virtual void SetWBBlue(float wb_blue) override;
  virtual float GetWBBlue() override;
  virtual CameraModeType GetMode() override;
  virtual void SetImageSizeAndMode(Shape shape, CameraModeType mode) override;
  virtual CameraPixelFormatType GetPixelFormat() override;
  virtual void SetPixelFormat(CameraPixelFormatType pixel_format) override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  CameraPixelFormatType VimbaPfmt2CameraPfmt(const string &vmb_pfmt);
  string CameraPfmt2VimbaPfmt(CameraPixelFormatType pfmt);

  void ResetDefaultCameraSettings();

  CameraPixelFormatType initial_pixel_format_;
  CameraModeType initial_mode_;

  VmbAPI::VimbaSystem &vimba_system_;
  VmbAPI::CameraPtr camera_;
};

#endif  // STREAMER_VIMBA_CAMERA_H
