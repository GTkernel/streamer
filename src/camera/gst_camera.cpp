//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include "gst_camera.h"

GSTCamera::GSTCamera(const string &name, const string &video_uri, int width,
                     int height)
    : Camera(name, video_uri, width, height) {}

bool GSTCamera::Init() {
  bool opened = capture_.CreatePipeline(video_uri_);

  if (!opened) {
    LOG(INFO) << "can't open camera";
    return false;
  }

  return true;
}
bool GSTCamera::OnStop() {
  capture_.DestroyPipeline();
  return true;
}
void GSTCamera::Process() {
  cv::Mat frame = capture_.GetFrame();
  PushFrame("bgr_output", new ImageFrame(frame, frame));
}
CameraType GSTCamera::GetCameraType() const { return CAMERA_TYPE_GST; }

// TODO: Implement camera control for GST camera, we may need to have subclass
// for IP camera, and use GstVideoCapture there.
float GSTCamera::GetExposure() { return 0; }
void GSTCamera::SetExposure(float exposure) {}
float GSTCamera::GetSharpness() { return 0; }
void GSTCamera::SetSharpness(float sharpness) {}
Shape GSTCamera::GetImageSize() { return nullptr; }
void GSTCamera::SetBrightness(float brightness) {}
float GSTCamera::GetBrightness() { return 0; }
void GSTCamera::SetShutterSpeed(float shutter_speed) {}
float GSTCamera::GetShutterSpeed() { return 0; }
void GSTCamera::SetSaturation(float saturation) {}
float GSTCamera::GetSaturation() { return 0; }
void GSTCamera::SetHue(float hue) {}
float GSTCamera::GetHue() { return 0; }
void GSTCamera::SetGain(float gain) {}
float GSTCamera::GetGain() { return 0; }
void GSTCamera::SetGamma(float gamma) {}
float GSTCamera::GetGamma() { return 0; }
void GSTCamera::SetWBRed(float wb_red) {}
float GSTCamera::GetWBRed() { return 0; }
void GSTCamera::SetWBBlue(float wb_blue) {}
float GSTCamera::GetWBBlue() { return 0; }
CameraModeType GSTCamera::GetMode() { return nullptr; }
void GSTCamera::SetImageSizeAndMode(Shape shape, CameraModeType mode) {}
CameraPixelFormatType GSTCamera::GetPixelFormat() { return nullptr; }
void GSTCamera::SetPixelFormat(CameraPixelFormatType pixel_format) {}
