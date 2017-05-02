//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#include "camera.h"
#include "utils/time_utils.h"

Camera::Camera(const string &name, const string &video_uri, int width,
               int height)
    : Processor({}, {"bgr_output"}),
      name_(name),
      video_uri_(video_uri),
      width_(width),
      height_(height) {
  stream_ = sinks_["bgr_output"];
}

string Camera::GetName() const { return name_; }

string Camera::GetVideoURI() const { return video_uri_; }

int Camera::GetWidth() { return width_; }
int Camera::GetHeight() { return height_; }

std::shared_ptr<Stream> Camera::GetStream() const { return stream_; }

ProcessorType Camera::GetType() { return PROCESSOR_TYPE_CAMERA; }

bool Camera::Capture(cv::Mat &image) {
  if (stopped_) {
    Start();
    auto reader = stream_->Subscribe();
    // Pop the first 3 images, the first few shots of the camera might be garbage
    for (int i = 0; i < 3; i++) {
      reader->PopFrame();
    }
    image = reader->PopFrame<ImageFrame>()->GetOriginalImage();
    reader->UnSubscribe();
    Stop();
  } else {
    auto reader = stream_->Subscribe();
    image = reader->PopFrame<ImageFrame>()->GetOriginalImage();
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

/*****************************************************************************
 * Pan/Tile, the implementation is ugly, but it is ok to show that we can
 * pan/tile the camera programmably.
 * TODO: A more general, and extendable way to unify this.
 *****************************************************************************/
void Camera::MoveUp() { int status = system((tile_up_command_ + " &").c_str()); }
void Camera::MoveDown() { int status = system((tile_down_command_ + " &").c_str()); }
void Camera::MoveLeft() { int status = system((pan_left_command_ + " &").c_str()); }
void Camera::MoveRight() { int status = system((pan_right_command_ + " &").c_str()); }