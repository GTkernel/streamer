//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"
#include <opencv2/core/core.hpp>

Frame::Frame(FrameType frame_type) : frame_type_(frame_type) {}
Frame::Frame(FrameType frame_type, cv::Mat original_image)
    : frame_type_(frame_type), original_image_(original_image) {}

cv::Mat Frame::GetOriginalImage() { return original_image_; }

void Frame::SetOriginalImage(cv::Mat original_image) {
  original_image_ = original_image;
}

ImageFrame::ImageFrame(cv::Mat image, cv::Mat original_image)
    : Frame(FRAME_TYPE_IMAGE, original_image),
      image_(image),
      shape_(image.channels(), image.cols, image.rows) {}

ImageFrame::ImageFrame(cv::Mat image)
    : Frame(FRAME_TYPE_IMAGE),
      image_(image),
      shape_(image.channels(), image.cols, image.rows) {}

Shape ImageFrame::GetSize() { return shape_; }

cv::Mat ImageFrame::GetImage() { return image_; }

void ImageFrame::SetImage(cv::Mat image) { image_ = image; }
FrameType Frame::GetType() { return frame_type_; }

MetadataFrame::MetadataFrame(string tag) : Frame(FRAME_TYPE_MD), tag_(tag) {}
MetadataFrame::MetadataFrame(float p1x, float p1y, float p2x, float p2y)
    : Frame(FRAME_TYPE_MD) {
  bbox_[0] = p1x;
  bbox_[1] = p1y;
  bbox_[2] = p2x;
  bbox_[3] = p2y;
}

MetadataFrame::MetadataFrame(string tag, cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image), tag_(tag) {}
MetadataFrame::MetadataFrame(float p1x, float p1y, float p2x, float p2y,
                             cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image) {
  bbox_[0] = p1x;
  bbox_[1] = p1y;
  bbox_[2] = p2x;
  bbox_[3] = p2y;
}
string MetadataFrame::GetTag() { return tag_; }
const float *MetadataFrame::GetBbox() { return bbox_; }
