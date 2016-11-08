//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "vimba_camera.h"
#include "utils/utils.h"

#include <VimbaImageTransform/Include/VmbTransform.h>

namespace VmbAPI = AVT::VmbAPI;

#define CHECK_VIMBA(cmd)                    \
  do {                                      \
    VmbErrorType error;                     \
    error = (cmd);                          \
    if (error != VmbErrorSuccess) {         \
      LOG(FATAL) << "VIMBA Error happened"; \
    }                                       \
  } while (0)

class VimbaCameraFrameObserver : public VmbAPI::IFrameObserver {
  friend class VimbaCamera;

 public:
  VimbaCameraFrameObserver(VimbaCamera *vimba_camera)
      : vimba_camera_(vimba_camera),
        VmbAPI::IFrameObserver(vimba_camera->camera_) {}

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
        if (!pFrame->GetBufferSize(buffer_size)) {
          LOG(ERROR) << "Can't get buffer size successfully";
        }

        LOG(INFO) << "Received image buffer of size: " << buffer_size;
        if (!pFrame->GetBuffer(vmb_buffer)) {
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
                         int height)
    : Camera(name, video_uri, width, height),
      vimba_system(VmbAPI::VimbaSystem::GetInstance()) {}

CameraType VimbaCamera::GetCameraType() const { return CAMERA_TYPE_VIMBA; }

bool VimbaCamera::Init() {
  string protocol, ip;
  ParseProtocolAndPath(video_uri_, protocol, ip);

  if (!vimba_system.OpenCameraByID(ip.c_str(), VmbAccessModeFull, camera_)) {
    LOG(ERROR) << "Can't open camera: " << name_;
    return false;
  } else {
    LOG(INFO) << "Vimba camera opened: " << name_;
  }

  // Now we have a Vimba camera handle
  camera_->StartContinuousImageAcquisition(
      10, VmbAPI::IFrameObserverPtr(new VimbaCameraFrameObserver(this)));
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

void VimbaCamera::SetGamma(float gamma) {}
float VimbaCamera::GetGamma() { return 0; }
void VimbaCamera::SetWBRed(float wb_red) {}
float VimbaCamera::GetWBRed() { return 0; }
void VimbaCamera::SetWBBlue(float wb_blue) {}
float VimbaCamera::GetWBBlue() { return 0; }
CameraModeType VimbaCamera::GetMode() { return CAMERA_MODE_0; }
void VimbaCamera::SetImageSizeAndMode(Shape shape, CameraModeType mode) {}
CameraPixelFormatType VimbaCamera::GetPixelFormat() {
  return CAMERA_PIXEL_FORMAT_INVALID;
}
void VimbaCamera::SetPixelFormat(CameraPixelFormatType pixel_format) {}

CameraPixelFormatType VimbaCamera::VimbaPfmt2CameraPfmt(
    VmbPixelFormatType vmb_pfmt) {
  return CAMERA_PIXEL_FORMAT_BGR;
}

VmbPixelFormatType VimbaCamera::CameraPfmt2VimbaPfmt(
    CameraPixelFormatType pfmt) {
  return VmbPixelFormatBayerBG8;
}
