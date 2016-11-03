//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include "pgr_camera.h"
#include "utils/utils.h"

#define CHECK_PGR(cmd)                       \
  do {                                       \
    FlyCapture2::Error error;                \
    error = (cmd);                           \
    if (error != FlyCapture2::PGRERROR_OK) { \
      error.PrintErrorTrace();               \
      LOG(FATAL) << "PGR Error happend";     \
    }                                        \
  } while (0)

PGRCamera::PGRCamera(const string &name, const string &video_uri, int width,
                     int height, FlyCapture2::Mode mode,
                     FlyCapture2::PixelFormat pixel_format)
    : Camera(name, video_uri, width, height, 2 /* # Sinks */),
      mode_(mode),
      pixel_format_(pixel_format) {}

bool PGRCamera::Init() {
  // Get the camera guid from ip address
  string protocol, ip;
  ParseProtocolAndPath(video_uri_, protocol, ip);
  FlyCapture2::BusManager bus_manager;

  unsigned int ip_addr = GetIPAddrFromString(ip);

  FlyCapture2::PGRGuid guid;
  CHECK_PGR(bus_manager.GetCameraFromIPAddress(ip_addr, &guid));
  CHECK_PGR(camera_.Connect(&guid));

  FlyCapture2::Format7ImageSettings fmt7ImageSettings;
  fmt7ImageSettings.mode = mode_;
  fmt7ImageSettings.offsetX = 0;
  fmt7ImageSettings.offsetY = 0;
  fmt7ImageSettings.width = (unsigned)width_;
  fmt7ImageSettings.height = (unsigned)height_;
  fmt7ImageSettings.pixelFormat = pixel_format_;

  bool valid;
  FlyCapture2::Format7PacketInfo fmt7PacketInfo;
  CHECK_PGR(camera_.ValidateFormat7Settings(&fmt7ImageSettings, &valid,
                                            &fmt7PacketInfo));
  CHECK_PGR(camera_.SetFormat7Configuration(
      &fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket));

  camera_.StartCapture();
  Reset();
  preprocess_thread_ = std::thread(&PGRCamera::PreProcess, this);

  LOG(INFO) << "Camera initialized";

  return true;
}

bool PGRCamera::OnStop() {
  camera_.StopCapture();
  camera_.Disconnect();
  preprocess_thread_.join();
  return true;
}

void PGRCamera::PreProcess() {
  while (!stopped_) {
    FlyCapture2::Image raw_image = preprocess_queue_.Pop();
    FlyCapture2::Image converted_image;
    DataBuffer image_bytes(raw_image.GetDataSize());
    image_bytes.Clone(raw_image.GetData(), raw_image.GetDataSize());
    raw_image.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &converted_image);

    unsigned int rowBytes =
        static_cast<unsigned>((double)converted_image.GetReceivedDataSize() /
                              (double)converted_image.GetRows());

    cv::Mat image =
        cv::Mat(converted_image.GetRows(), converted_image.GetCols(), CV_8UC3,
                converted_image.GetData(), rowBytes);

    cv::Mat output_image;
    output_image = image.clone();

    PushFrame(0, new ImageFrame(output_image, output_image));
    PushFrame(1, new BytesFrame(image_bytes, output_image));
  }
}

void PGRCamera::Process() {
  Timer timer, retrieve_timer;
  timer.Start();
  FlyCapture2::Image raw_image;
  FlyCapture2::Error error;

  {
    std::lock_guard<std::mutex> guard(camera_lock_);
    retrieve_timer.Start();
    error = camera_.RetrieveBuffer(&raw_image);
  }

  if (error != FlyCapture2::PGRERROR_OK) {
    //    error.PrintErrorTrace();
    return;
  }

  //  LOG(INFO) << "Retrieve took " << retrieve_timer.ElapsedMSec() << " ms";

  preprocess_queue_.Push(raw_image);
  //  LOG(INFO) << "Process took " << timer.ElapsedMSec() << " ms";
}

float PGRCamera::GetExposure() {
  return GetProperty(FlyCapture2::AUTO_EXPOSURE, true, false);
}
void PGRCamera::SetExposure(float exposure) {
  SetProperty(FlyCapture2::AUTO_EXPOSURE, exposure, true, false);
}
float PGRCamera::GetSharpness() {
  return GetProperty(FlyCapture2::SHARPNESS, false, true);
}
void PGRCamera::SetSharpness(float sharpness) {
  SetProperty(FlyCapture2::SHARPNESS, sharpness, false, true);
}

void PGRCamera::SetBrightness(float brightness) {
  brightness = std::max(0.0f, brightness);
  SetProperty(FlyCapture2::BRIGHTNESS, brightness, true, false);
}

float PGRCamera::GetBrightness() {
  return GetProperty(FlyCapture2::BRIGHTNESS, true, false);
}

void PGRCamera::SetShutterSpeed(float shutter_speed) {
  shutter_speed = std::max(5.0f, shutter_speed);
  SetProperty(FlyCapture2::SHUTTER, shutter_speed, true, false);
}
float PGRCamera::GetShutterSpeed() {
  GetProperty(FlyCapture2::SHUTTER, true, false);
}

void PGRCamera::SetSaturation(float saturation) {
  SetProperty(FlyCapture2::SATURATION, saturation, true, false);
}

float PGRCamera::GetSaturation() {
  GetProperty(FlyCapture2::SATURATION, true, false);
}

void PGRCamera::SetHue(float hue) {
  SetProperty(FlyCapture2::HUE, hue, true, false);
}
float PGRCamera::GetHue() { GetProperty(FlyCapture2::HUE, true, false); }

void PGRCamera::SetGain(float gain) {
  SetProperty(FlyCapture2::GAIN, gain, true, false);
}

float PGRCamera::GetGain() {
  return GetProperty(FlyCapture2::GAIN, true, false);
}
void PGRCamera::SetGamma(float gamma) {
  SetProperty(FlyCapture2::GAMMA, gamma, true, false);
}
float PGRCamera::GetGamma() {
  return GetProperty(FlyCapture2::GAMMA, true, false);
}
void PGRCamera::SetWBRed(float wb_red) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::WHITE_BALANCE;
  prop.onOff = true;
  prop.autoManualMode = false;
  prop.absControl = false;
  prop.valueA = (unsigned int)wb_red;
  prop.valueB = (unsigned int)GetWBBlue();

  CHECK_PGR(camera_.SetProperty(&prop));
}
float PGRCamera::GetWBRed() {
  return GetProperty(FlyCapture2::WHITE_BALANCE, false, true);
}
void PGRCamera::SetWBBlue(float wb_blue) {
  FlyCapture2::Property prop;
  prop.type = FlyCapture2::WHITE_BALANCE;
  prop.onOff = true;
  prop.autoManualMode = false;
  prop.absControl = false;
  prop.valueA = (unsigned int)GetWBRed();
  prop.valueB = (unsigned int)wb_blue;

  CHECK_PGR(camera_.SetProperty(&prop));
}

float PGRCamera::GetWBBlue() {
  return GetProperty(FlyCapture2::WHITE_BALANCE, false, false);
}

FlyCapture2::PixelFormat PGRCamera::GetPixelFormat() { return pixel_format_; }

Shape PGRCamera::GetImageSize() {
  FlyCapture2::Format7ImageSettings image_settings;
  unsigned int current_packet_size;
  float current_percentage;
  CHECK_PGR(camera_.GetFormat7Configuration(
      &image_settings, &current_packet_size, &current_percentage));
  return Shape(image_settings.width, image_settings.height);
}

FlyCapture2::VideoMode PGRCamera::GetVideoMode() {
  FlyCapture2::VideoMode video_mode;
  FlyCapture2::FrameRate frame_rate;
  camera_.GetVideoModeAndFrameRate(&video_mode, &frame_rate);

  return video_mode;
}

void PGRCamera::SetImageSizeAndVideoMode(Shape shape, FlyCapture2::Mode mode) {
  std::lock_guard<std::mutex> guard(camera_lock_);
  CHECK_PGR(camera_.StopCapture());

  // Get fmt7 image settings
  FlyCapture2::Format7ImageSettings image_settings;
  unsigned int current_packet_size;
  float current_percentage;
  CHECK_PGR(camera_.GetFormat7Configuration(
      &image_settings, &current_packet_size, &current_percentage));

  image_settings.mode = mode;
  image_settings.height = (unsigned)shape.height;
  image_settings.width = (unsigned)shape.width;
  bool valid;
  FlyCapture2::Format7PacketInfo fmt7_packet_info;

  CHECK_PGR(camera_.ValidateFormat7Settings(&image_settings, &valid,
                                            &fmt7_packet_info));
  CHECK(valid) << "fmt7 image settings are not valid";

  CHECK_PGR(camera_.SetFormat7Configuration(
      &image_settings, fmt7_packet_info.recommendedBytesPerPacket));
  CHECK_PGR(camera_.StartCapture());
}

void PGRCamera::SetPixelFormat(FlyCapture2::PixelFormat pixel_format) {
  std::lock_guard<std::mutex> guard(camera_lock_);
  pixel_format_ = pixel_format;
  CHECK_PGR(camera_.StopCapture());

  // Get fmt7 image settings
  FlyCapture2::Format7ImageSettings image_settings;
  unsigned int current_packet_size;
  float current_percentage;
  CHECK_PGR(camera_.GetFormat7Configuration(
      &image_settings, &current_packet_size, &current_percentage));

  image_settings.pixelFormat = pixel_format_;
  bool valid;
  FlyCapture2::Format7PacketInfo fmt7_packet_info;

  CHECK_PGR(camera_.ValidateFormat7Settings(&image_settings, &valid,
                                            &fmt7_packet_info));
  CHECK(valid) << "fmt7 image settings are not valid";

  CHECK_PGR(camera_.SetFormat7Configuration(
      &image_settings, fmt7_packet_info.recommendedBytesPerPacket));
  CHECK_PGR(camera_.StartCapture());
}

void PGRCamera::SetProperty(FlyCapture2::PropertyType property_type,
                            float value, bool abs, bool value_a) {
  FlyCapture2::Property prop;
  prop.type = property_type;
  prop.onOff = true;
  prop.autoManualMode = false;

  if (!abs) {
    prop.absControl = false;
    if (value_a) {
      prop.valueA = (unsigned)value;
    } else {
      prop.valueB = (unsigned)value;
    }
  } else {
    prop.absControl = true;
    prop.absValue = value;
  }

  CHECK_PGR(camera_.SetProperty(&prop));
}

float PGRCamera::GetProperty(FlyCapture2::PropertyType property_type, bool abs,
                             bool value_a) {
  //  LOG(INFO) << "Get property called";
  FlyCapture2::Property prop;
  prop.type = property_type;
  CHECK_PGR(camera_.GetProperty(&prop));

  if (abs) {
    return prop.absValue;
  } else {
    if (value_a) {
      return (float)prop.valueA;
    } else {
      return (float)prop.valueB;
    }
  }
}
CameraType PGRCamera::GetType() const { return CAMERA_TYPE_PTGRAY; }

void PGRCamera::Reset() {
  FlyCapture2::Property prop;
  prop.onOff = true;
  prop.autoManualMode = true;

  prop.type = FlyCapture2::BRIGHTNESS;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::SHARPNESS;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::GAMMA;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::GAIN;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::AUTO_EXPOSURE;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::WHITE_BALANCE;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::SATURATION;
  CHECK_PGR(camera_.SetProperty(&prop));

  prop.type = FlyCapture2::HUE;
  CHECK_PGR(camera_.SetProperty(&prop));
}
