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
      : vimba_camera_(vimba_camera),
        VmbAPI::IFrameObserver(vimba_camera->camera_) {}

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

        DataBuffer data_buffer(buffer_size);
        data_buffer.Clone(DataBuffer(vmb_buffer, buffer_size));

        vimba_camera_->PushFrame("raw_output", new BytesFrame(data_buffer));
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
