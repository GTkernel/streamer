//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//
#include "gst_camera.h"

#include <chrono>

GSTCamera::GSTCamera(const string& name, const string& video_uri, int width,
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
  const cv::Mat& pixels = capture_.GetPixels();
  auto frame = std::make_unique<Frame>();
  frame->SetValue("original_bytes",
                  std::vector<char>(
                      (char*)pixels.data,
                      (char*)pixels.data + pixels.total() * pixels.elemSize()));
  frame->SetValue("original_image", pixels);
  MetadataToFrame(frame);
  PushFrame("output", std::move(frame));
}

CameraType GSTCamera::GetCameraType() const { return CAMERA_TYPE_GST; }

// TODO: Implement camera control for GST camera, we may need to have subclass
// for IP camera, and use GstVideoCapture there.
float GSTCamera::GetExposure() { return 0; }
void GSTCamera::SetExposure(float) {}
float GSTCamera::GetSharpness() { return 0; }
void GSTCamera::SetSharpness(float) {}
Shape GSTCamera::GetImageSize() { return Shape(); }
void GSTCamera::SetBrightness(float) {}
float GSTCamera::GetBrightness() { return 0; }
void GSTCamera::SetSaturation(float) {}
float GSTCamera::GetSaturation() { return 0; }
void GSTCamera::SetHue(float) {}
float GSTCamera::GetHue() { return 0; }
void GSTCamera::SetGain(float) {}
float GSTCamera::GetGain() { return 0; }
void GSTCamera::SetGamma(float) {}
float GSTCamera::GetGamma() { return 0; }
void GSTCamera::SetWBRed(float) {}
float GSTCamera::GetWBRed() { return 0; }
void GSTCamera::SetWBBlue(float) {}
float GSTCamera::GetWBBlue() { return 0; }
CameraModeType GSTCamera::GetMode() { return CAMERA_MODE_INVALID; }
void GSTCamera::SetImageSizeAndMode(Shape, CameraModeType) {}
CameraPixelFormatType GSTCamera::GetPixelFormat() {
  return CAMERA_PIXEL_FORMAT_INVALID;
}
void GSTCamera::SetPixelFormat(CameraPixelFormatType) {}
void GSTCamera::SetFrameRate(float) {}
float GSTCamera::GetFrameRate() { return 0; }
void GSTCamera::SetROI(int, int, int,
                       int) {}
int GSTCamera::GetROIOffsetX() { return 0; }
int GSTCamera::GetROIOffsetY() { return 0; }
Shape GSTCamera::GetROIOffsetShape() { return Shape(); }
