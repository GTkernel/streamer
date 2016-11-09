//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "vimba_camera.h"
#include "utils/utils.h"

namespace VmbAPI = AVT::VmbAPI;

class VimbaCameraFrameObserver : public VmbAPI::IFrameObserver {
  friend class VimbaCamera;

 public:
  VimbaCameraFrameObserver(VimbaCamera *vimba_camera)
      : VmbAPI::IFrameObserver(vimba_camera->camera_),
        vimba_camera_(vimba_camera) {}

  /**
   * @brief Transfrom the Vimba image into a BGR OpenCV Mat.
   * @param pFrame The captured frame.
   * @return The Mat of the transformed BGR image.
   */
  cv::Mat TransformToBGRImage(const VmbAPI::FramePtr pFrame) {
    VmbImage sourceImage;
    VmbImage destinationImage;
    VmbTransformInfo info;
    VmbUint32_t vmb_width, vmb_height;
    CHECK_VIMBA(pFrame->GetWidth(vmb_width));
    CHECK_VIMBA(pFrame->GetHeight(vmb_height));

    size_t width = vmb_width, height = vmb_height;

    LOG(INFO) << "Received image with width: " << width << " " << height;

    cv::Mat dest_bgr_mat((int)height, (int)width, CV_8UC3);

    // set size member for verification inside API
    sourceImage.Size = sizeof(sourceImage);
    destinationImage.Size = sizeof(destinationImage);

    // attach the data buffers
    VmbUchar_t *input_buffer;
    VmbUchar_t *output_buffer = dest_bgr_mat.data;
    CHECK_VIMBA(pFrame->GetBuffer(input_buffer));

    sourceImage.Data = input_buffer;
    destinationImage.Data = output_buffer;

    VmbPixelFormatType input_pfmt;
    CHECK_VIMBA(pFrame->GetPixelFormat(input_pfmt));

    VmbSetImageInfoFromPixelFormat(input_pfmt, vmb_width, vmb_height,
                                   &sourceImage);

    VmbSetImageInfoFromInputImage(&sourceImage, VmbPixelLayoutBGR, 8,
                                  &destinationImage);

    VmbSetDebayerMode(VmbDebayerMode2x2, &info);

    // Perform the transformation
    VmbImageTransform(&sourceImage, &destinationImage, &info, 1);

    return dest_bgr_mat;
  }

  void FrameReceived(const VmbAPI::FramePtr pFrame) {
    VmbFrameStatusType eReceiveStatus;
    if (VmbErrorSuccess == pFrame->GetReceiveStatus(eReceiveStatus)) {
      if (VmbFrameStatusComplete == eReceiveStatus) {
        // Put your code here to react on a successfully received frame
        // Copy the data of the frame

        VmbUint32_t buffer_size;
        VmbUchar_t *vmb_buffer;
        // We don't use CHECK_VIMBA here because we don't want to crash for
        // unsuccessful image grab
        if (VmbErrorSuccess != pFrame->GetBufferSize(buffer_size)) {
          LOG(ERROR) << "Can't get buffer size successfully";
        }

        if (VmbErrorSuccess != pFrame->GetBuffer(vmb_buffer)) {
          LOG(ERROR) << "Can't get vimba buffer";
        }

        // Raw bytes of the image
        DataBuffer data_buffer(buffer_size);
        data_buffer.Clone(DataBuffer(vmb_buffer, buffer_size));

        // Transform to BGR image
        cv::Mat bgr_output = TransformToBGRImage(pFrame);

        vimba_camera_->PushFrame("raw_output",
                                 new BytesFrame(data_buffer, bgr_output));
        vimba_camera_->PushFrame("bgr_output",
                                 new ImageFrame(bgr_output, bgr_output));
      } else {
        LOG(ERROR) << "Can't get frame successfully: " << eReceiveStatus;
      }  // Validate eReceiveStatus
    }    // GetReceiveStatus
    m_pCamera->QueueFrame(pFrame);
  }

 private:
  VimbaCamera *vimba_camera_;
};

VimbaCamera::VimbaCamera(const string &name, const string &video_uri, int width,
                         int height, CameraModeType mode,
                         CameraPixelFormatType pixel_format)
    : Camera(name, video_uri, width, height),
      initial_pixel_format_(pixel_format),
      initial_mode_(mode),
      vimba_system_(VmbAPI::VimbaSystem::GetInstance()) {
  // Init raw output sink
  sinks_.insert({"raw_output", StreamPtr(new Stream)});
}

CameraType VimbaCamera::GetCameraType() const { return CAMERA_TYPE_VIMBA; }

bool VimbaCamera::Init() {
  string protocol, ip;
  ParseProtocolAndPath(video_uri_, protocol, ip);

  if (StringContains(ip, ".")) {
    // Looks like an IP
    CHECK_VIMBA(
        vimba_system_.OpenCameraByID(ip.c_str(), VmbAccessModeFull, camera_));
  } else {
    // Looks like an device index
    int device_idx = StringToInt(ip);
    VmbAPI::CameraPtrVector cameras;
    CHECK_VIMBA(vimba_system_.GetCameras(cameras));
    CHECK(device_idx < cameras.size()) << "Invalid camera index: "
                                       << device_idx;
    camera_ = cameras[device_idx];
    camera_->Open(VmbAccessModeFull);
  }

  // Now we have a Vimba camera handle
  camera_->StartContinuousImageAcquisition(
      11, VmbAPI::IFrameObserverPtr(new VimbaCameraFrameObserver(this)));

  // Reset to default camera settings
  ResetDefaultCameraSettings();

  return true;
}

void VimbaCamera::ResetDefaultCameraSettings() {
  SetImageSizeAndMode(Shape(width_, height_), initial_mode_);
  SetPixelFormat(initial_pixel_format_);
}

bool VimbaCamera::OnStop() {
  if (VmbErrorSuccess == camera_->Close()) {
    LOG(INFO) << "Camera closed";
  } else {
    LOG(INFO) << "Can't close camera: " << name_;
  }

  camera_->StopContinuousImageAcquisition();
}

void VimbaCamera::Process() {
  // Process will do nothing as we are asynchronously receive image frames.
}

// TODO: Implement camera controls
float VimbaCamera::GetExposure() {
  VmbAPI::FeaturePtr pFeature;
  double exposure;
  CHECK_VIMBA(camera_->GetFeatureByName("ExposureTimeAbs", pFeature));
  CHECK_VIMBA(pFeature->GetValue(exposure));
  return (float)exposure;
}

void VimbaCamera::SetExposure(float exposure) {
  VmbAPI::FeaturePtr pFeature;
  CHECK_VIMBA(camera_->GetFeatureByName("ExposureTimeAbs", pFeature));
  CHECK_VIMBA(pFeature->SetValue((double)exposure));
}

float VimbaCamera::GetSharpness() { return 0; }
void VimbaCamera::SetSharpness(float sharpness) {}
Shape VimbaCamera::GetImageSize() { return Shape(); }
void VimbaCamera::SetBrightness(float brightness) {}
float VimbaCamera::GetBrightness() { return 0; }
void VimbaCamera::SetShutterSpeed(float shutter_speed) {}
float VimbaCamera::GetShutterSpeed() { return 0; }
void VimbaCamera::SetSaturation(float saturation) {}
float VimbaCamera::GetSaturation() { return 0; }
void VimbaCamera::SetHue(float hue) {}
float VimbaCamera::GetHue() { return 0; }

void VimbaCamera::SetGain(float gain) {
  VmbAPI::FeaturePtr pFeature;
  CHECK_VIMBA(camera_->GetFeatureByName("Gain", pFeature));
  CHECK_VIMBA(pFeature->SetValue(gain));
}
float VimbaCamera::GetGain() {
  VmbAPI::FeaturePtr pFeature;
  double gain;

  CHECK_VIMBA(camera_->GetFeatureByName("Gain", pFeature));
  CHECK_VIMBA(pFeature->GetValue(gain));

  return (float)gain;
}

void VimbaCamera::SetGamma(float gamma) {
  VmbAPI::FeaturePtr pFeature;

  CHECK_VIMBA(camera_->GetFeatureByName("Gamma", pFeature));
  CHECK_VIMBA(pFeature->SetValue(gamma));
}

float VimbaCamera::GetGamma() {
  VmbAPI::FeaturePtr pFeature;
  double gamma;

  CHECK_VIMBA(camera_->GetFeatureByName("Gamma", pFeature));
  CHECK_VIMBA(pFeature->GetValue(gamma));

  return (float)gamma;
}

void VimbaCamera::SetWBRed(float wb_red) {}

float VimbaCamera::GetWBRed() { return 0; }

void VimbaCamera::SetWBBlue(float wb_blue) {}

float VimbaCamera::GetWBBlue() { return 0; }

CameraModeType VimbaCamera::GetMode() {
  VmbAPI::FeaturePtr pFeature;
  VmbInt64_t binning;

  CHECK_VIMBA(camera_->GetFeatureByName("BinningHorizontal", pFeature));
  CHECK_VIMBA(pFeature->GetValue(binning));

  if (binning == 1) {
    return CAMERA_MODE_0;
  } else if (binning == 2) {
    return CAMERA_MODE_1;
  } else if (binning == 4) {
    return CAMERA_MODE_2;
  } else if (binning == 8) {
    return CAMERA_MODE_3;
  }

  return CAMERA_MODE_INVALID;
}

void VimbaCamera::SetImageSizeAndMode(Shape shape, CameraModeType mode) {
  VmbAPI::FeaturePtr pFeature;
  VmbInt64_t binning, width, height;

  CHECK(mode != CAMERA_MODE_INVALID);
  if (mode == 0) {
    binning = 1;
  } else if (mode == 1) {
    binning = 2;
  } else if (mode == 2) {
    binning = 4;
  } else if (mode == 3) {
    binning = 8;
  }

  CHECK_VIMBA(camera_->GetFeatureByName("BinningHorizontal", pFeature));
  CHECK_VIMBA(pFeature->SetValue(binning));
  CHECK_VIMBA(camera_->GetFeatureByName("BinningVertical", pFeature));
  CHECK_VIMBA(pFeature->SetValue(binning));
  CHECK_VIMBA(camera_->GetFeatureByName("Width", pFeature));
  pFeature->SetValue(shape.width);
  CHECK_VIMBA(camera_->GetFeatureByName("Height", pFeature));
  pFeature->SetValue(shape.height);
}

CameraPixelFormatType VimbaCamera::GetPixelFormat() {
  VmbAPI::FeaturePtr pFeature;
  string vimba_pfmt;

  CHECK_VIMBA(camera_->GetFeatureByName("PixelFormat", pFeature));
  CHECK_VIMBA(pFeature->GetValue(vimba_pfmt));

  return VimbaPfmt2CameraPfmt(vimba_pfmt);
}

void VimbaCamera::SetPixelFormat(CameraPixelFormatType pixel_format) {
  VmbAPI::FeaturePtr pFeature;
  CHECK_VIMBA(camera_->GetFeatureByName("PixelFormat", pFeature));
  CHECK_VIMBA(pFeature->SetValue(CameraPfmt2VimbaPfmt(pixel_format).c_str()));
}

CameraPixelFormatType VimbaCamera::VimbaPfmt2CameraPfmt(
    const string &vmb_pfmt) {
  if (vmb_pfmt == "Mono8") {
    return CAMERA_PIXEL_FORMAT_MONO8;
  } else if (vmb_pfmt == "BayerRG8") {
    return CAMERA_PIXEL_FORMAT_RAW8;
  } else if (vmb_pfmt == "BayerRG12") {
    return CAMERA_PIXEL_FORMAT_RAW12;
  } else if (vmb_pfmt == "BGR8Packed") {
    return CAMERA_PIXEL_FORMAT_BGR;
  } else if (vmb_pfmt == "YUV411Packed") {
    return CAMERA_PIXEL_FORMAT_YUV411;
  } else if (vmb_pfmt == "YUV422Packed") {
    return CAMERA_PIXEL_FORMAT_YUV422;
  } else if (vmb_pfmt == "YUV444Packed") {
    return CAMERA_PIXEL_FORMAT_YUV444;
  } else {
    LOG(FATAL) << "Invalid or unsupported Vimba pixel format: " << vmb_pfmt;
  }

  return CAMERA_PIXEL_FORMAT_MONO8;
}

string VimbaCamera::CameraPfmt2VimbaPfmt(CameraPixelFormatType pfmt) {
  switch (pfmt) {
    // TODO: Not very sure about the pixel format mapping, if something wrong
    // with color convertion happens, check here.
    case CAMERA_PIXEL_FORMAT_MONO8:
      return "Mono8";
    case CAMERA_PIXEL_FORMAT_RAW8:
      return "BayerRG8";
    case CAMERA_PIXEL_FORMAT_RAW12:
      return "BayerRG12";
    case CAMERA_PIXEL_FORMAT_BGR:
      return "BGR8Packed";
    case CAMERA_PIXEL_FORMAT_YUV411:
      return "YUV411Packed";
    case CAMERA_PIXEL_FORMAT_YUV422:
      return "YUV422Packed";
    case CAMERA_PIXEL_FORMAT_YUV444:
      return "YUV444Packed";
    default:
      LOG(FATAL) << "Invalid pixel format: " << pfmt;
  }

  return "Mono8";
}
