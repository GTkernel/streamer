//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#ifndef STREAMER_PGR_CAMERA_H
#define STREAMER_PGR_CAMERA_H

#include <flycapture/FlyCapture2.h>

#include "camera.h"
#include "utils/utils.h"

/**
 * @brief A class for ptgray camera, in order to use this class, you have to
 * make sure that the SDK for ptgray camera is installed on the system.
 */
class PGRCamera : public Camera {
 public:
  PGRCamera(const string& name, const string& video_uri, int width = -1,
            int height = -1, CameraModeType mode = CAMERA_MODE_0,
            CameraPixelFormatType pixel_format = CAMERA_PIXEL_FORMAT_RAW12);
  virtual CameraType GetCameraType() const override;

  // Camera controls
  virtual float GetExposure();
  virtual void SetExposure(float exposure);
  virtual float GetSharpness();
  virtual void SetSharpness(float sharpness);
  virtual Shape GetImageSize();
  virtual void SetBrightness(float brightness);
  virtual float GetBrightness();
  virtual void SetSaturation(float saturation);
  virtual float GetSaturation();
  virtual void SetHue(float hue);
  virtual float GetHue();
  virtual void SetGain(float gain);
  virtual float GetGain();
  virtual void SetGamma(float gamma);
  virtual float GetGamma();
  virtual void SetWBRed(float wb_red);
  virtual float GetWBRed();
  virtual void SetWBBlue(float wb_blue);
  virtual float GetWBBlue();
  virtual CameraModeType GetMode();
  virtual void SetImageSizeAndMode(Shape shape, CameraModeType mode);
  virtual CameraPixelFormatType GetPixelFormat();
  virtual void SetPixelFormat(CameraPixelFormatType pixel_format);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;
  void Reset();

 private:
  /**
   * @brief Set the property of the camera in either int value or abs value.
   * @param property_type The type of the property.
   * @param value The value of the property.
   * @param abs Wether the value should be set as abs value (float) or not.
   * @param int_value_idx Set value_a or value_b.
   */
  void SetProperty(FlyCapture2::PropertyType property_type, float value,
                   bool abs, bool value_a = true);
  float GetProperty(FlyCapture2::PropertyType property_type, bool abs,
                    bool value_a = true);
  FlyCapture2::Format7ImageSettings GetImageSettings();

  static void OnImageGrabbed(FlyCapture2::Image* image, const void* user_data);

  CameraModeType FCMode2CameraMode(FlyCapture2::Mode fc_mode);
  FlyCapture2::Mode CameraMode2FCMode(CameraModeType mode);
  CameraPixelFormatType FCPfmt2CameraPfmt(FlyCapture2::PixelFormat fc_pfmt);
  FlyCapture2::PixelFormat CameraPfmt2FCPfmt(CameraPixelFormatType pfmt);

  CameraPixelFormatType initial_pixel_format_;
  CameraModeType initial_mode_;

  FlyCapture2::Camera camera_;
  std::mutex camera_lock_;
};

#endif  // STREAMER_PGR_CAMERA_H
