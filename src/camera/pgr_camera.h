//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#ifndef STREAMER_PGR_CAMERA_H
#define STREAMER_PGR_CAMERA_H

#include <flycapture/FlyCapture2.h>
#include "camera.h"

/**
 * @brief A class for ptgray camera, in order to use this class, you have to
 * make sure that the SDK for ptgray camera is installed on the system.
 */
class PGRCamera : public Camera {
 public:
  PGRCamera(const string &name, const string &video_uri, int width = -1,
            int height = -1, FlyCapture2::Mode mode = FlyCapture2::MODE_0,
            FlyCapture2::PixelFormat pixel_format =
                FlyCapture2::PIXEL_FORMAT_411YUV8);
  virtual CameraType GetType() const override;

  float GetExposure();
  void SetExposure(float exposure);
  float GetSharpness();
  void SetSharpness(float sharpness);
  Shape GetImageSize();
  FlyCapture2::VideoMode GetVideoMode();
  void SetImageSizeAndVideoMode(Shape shape, FlyCapture2::Mode mode);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

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

  FlyCapture2::Camera camera_;
  FlyCapture2::Mode mode_;
  FlyCapture2::PixelFormat pixel_format_;

  std::mutex camera_lock_;
};

#endif  // STREAMER_PGR_CAMERA_H
