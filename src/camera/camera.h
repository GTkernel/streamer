//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#ifndef STREAMER_CAMERA_H
#define STREAMER_CAMERA_H

#include "common/common.h"
#include "processor/processor.h"
#include "stream/stream.h"

/**
 * @brief This class represents a camera available on the device.
 */
class Camera : public Processor {
 public:
  Camera(const string &name, const string &video_uri, int width = -1,
         int height = -1);  // Just a nonsense default value
  string GetName() const;
  string GetVideoURI() const;
  std::shared_ptr<Stream> GetStream() const;
  int GetWidth();
  int GetHeight();

  virtual bool Capture(cv::Mat &image);
  virtual CameraType GetCameraType() const = 0;
  virtual ProcessorType GetType() override;

  // Camera controls
  virtual float GetExposure() = 0;
  virtual void SetExposure(float exposure) = 0;
  virtual float GetSharpness() = 0;
  virtual void SetSharpness(float sharpness) = 0;
  virtual Shape GetImageSize() = 0;
  virtual void SetBrightness(float brightness) = 0;
  virtual float GetBrightness() = 0;
  virtual void SetSaturation(float saturation) = 0;
  virtual float GetSaturation() = 0;
  virtual void SetHue(float hue) = 0;
  virtual float GetHue() = 0;
  virtual void SetGain(float gain) = 0;
  virtual float GetGain() = 0;
  virtual void SetGamma(float gamma) = 0;
  virtual float GetGamma() = 0;
  virtual void SetWBRed(float wb_red) = 0;
  virtual float GetWBRed() = 0;
  virtual void SetWBBlue(float wb_blue) = 0;
  virtual float GetWBBlue() = 0;
  virtual CameraModeType GetMode() = 0;
  virtual void SetImageSizeAndMode(Shape shape, CameraModeType mode) = 0;
  virtual CameraPixelFormatType GetPixelFormat() = 0;
  virtual void SetPixelFormat(CameraPixelFormatType pixel_format) = 0;

  string GetCameraInfo();

 protected:
  virtual bool Init() override = 0;
  virtual bool OnStop() override = 0;
  virtual void Process() override = 0;

 protected:
  string name_;
  string video_uri_;
  int width_;
  int height_;
  // Camera output stream
  std::shared_ptr<Stream> stream_;
};

#endif  // STREAMER_CAMERA_H
