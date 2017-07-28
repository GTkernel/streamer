//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera.h"

#include "common/types.h"
#include "utils/time_utils.h"
#include "utils/utils.h"

Camera::Camera(const string& name, const string& video_uri, int width,
               int height)
    : Processor(PROCESSOR_TYPE_CAMERA, {}, {"output"}),
      name_(name),
      video_uri_(video_uri),
      width_(width),
      height_(height),
      frame_id_(0) {
  stream_ = sinks_["output"];
}

string Camera::GetName() const { return name_; }

string Camera::GetVideoURI() const { return video_uri_; }

int Camera::GetWidth() { return width_; }
int Camera::GetHeight() { return height_; }

std::shared_ptr<Stream> Camera::GetStream() const { return stream_; }

unsigned long Camera::CreateFrameID() { return frame_id_++; }

bool Camera::Capture(cv::Mat& image) {
  if (stopped_) {
    LOG(WARNING) << "stopped.";
    Start();
    auto reader = stream_->Subscribe();
    // Pop the first 3 images, the first few shots of the camera might be
    // garbage
    for (int i = 0; i < 3; i++) {
      reader->PopFrame();
    }
    image = reader->PopFrame()->GetValue<cv::Mat>("original_image");
    reader->UnSubscribe();
    Stop();
  } else {
    LOG(WARNING) << "not stopped.";
    auto reader = stream_->Subscribe();
    image = reader->PopFrame()->GetValue<cv::Mat>("original_image");
    reader->UnSubscribe();
  }

  return true;
}

/**
 * Get the camera parameters information.
 */
string Camera::GetCameraInfo() {
  std::ostringstream ss;
  ss << "name: " << GetName() << "\n";
  ss << "record time: " << GetCurrentTimeString("%Y%m%d-%H%M%S") << "\n";
  ss << "image size: " << GetImageSize().width << "x" << GetImageSize().height
     << "\n";
  ss << "pixel format: " << GetCameraPixelFormatString(GetPixelFormat())
     << "\n";
  ss << "exposure: " << GetExposure() << "\n";
  ss << "gain: " << GetGain() << "\n";
  return ss.str();
}

void Camera::MetadataToFrame(std::unique_ptr<Frame>& frame) {
  frame->SetValue<float>("CameraSettings.Exposure", GetExposure());
  frame->SetValue<float>("CameraSettings.Sharpness", GetSharpness());
  frame->SetValue<float>("CameraSettings.Brightness", GetBrightness());
  frame->SetValue<float>("CameraSettings.Saturation", GetSaturation());
  frame->SetValue<float>("CameraSettings.Hue", GetHue());
  frame->SetValue<float>("CameraSettings.Gain", GetGain());
  frame->SetValue<float>("CameraSettings.Gamma", GetGamma());
  frame->SetValue<float>("CameraSettings.WBRed", GetWBRed());
  frame->SetValue<float>("CameraSettings.WBBlue", GetWBBlue());
}

/*****************************************************************************
 * Pan/Tile, the implementation is ugly, but it is ok to show that we can
 * pan/tile the camera programmably.
 * TODO: A more general, and extendable way to unify this.
 *****************************************************************************/
void Camera::MoveUp() { ExecuteAndCheck(tile_up_command_ + " &"); }
void Camera::MoveDown() { ExecuteAndCheck((tile_down_command_ + " &")); }
void Camera::MoveLeft() { ExecuteAndCheck((pan_left_command_ + " &")); }
void Camera::MoveRight() { ExecuteAndCheck((pan_right_command_ + " &")); }
