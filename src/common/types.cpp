//
// Created by Ran Xian (xranthoar@gmail.com) on 9/2/16.
//

#include "types.h"

std::string GetCameraPixelFormatString(CameraPixelFormatType pfmt) {
  switch (pfmt) {
    case CAMERA_PIXEL_FORMAT_RAW8:
      return "RAW8";
    case CAMERA_PIXEL_FORMAT_RAW12:
      return "RAW12";
    case CAMERA_PIXEL_FORMAT_MONO8:
      return "Mono8";
    case CAMERA_PIXEL_FORMAT_BGR:
      return "BGR";
    case CAMERA_PIXEL_FORMAT_YUV411:
      return "YUV411";
    case CAMERA_PIXEL_FORMAT_YUV422:
      return "YUV422";
    case CAMERA_PIXEL_FORMAT_YUV444:
      return "YUV444";
    default:
      return "PIXEL_FORMAT_INVALID";
  }
}
