//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"
#include <opencv2/core/core.hpp>

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

Shape ImageFrame::GetSize() { return shape_; }

cv::Mat ImageFrame::GetImage() { return image_; }

void ImageFrame::SetImage(cv::Mat image) { image_ = image; }
FrameType ImageFrame::GetType() { return FRAME_TYPE_IMAGE; }
FrameType Frame::GetType() { return frame_type_; }

MetadataFrame::MetadataFrame(std::vector<string> tags, cv::Mat original_image)
    : Frame(FRAME_TYPE_IMAGE, original_image), tags_(tags) {
  bitset_.set(Bit_tags);
}
MetadataFrame::MetadataFrame(std::vector<Rect> bboxes, cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image), bboxes_(bboxes) {
  bitset_.set(Bit_bboxes);
}
MetadataFrame::MetadataFrame(std::vector<string> tags, std::vector<Rect> bboxes, cv::Mat original_image)
    : Frame(FRAME_TYPE_MD, original_image), tags_(tags), bboxes_(bboxes) {
  bitset_.set(Bit_tags);
  bitset_.set(Bit_bboxes);
}

std::vector<string> MetadataFrame::GetTags() { return tags_; }
std::vector<Rect> MetadataFrame::GetBboxes() { return bboxes_; }
FrameType MetadataFrame::GetType() { return FRAME_TYPE_MD; }

BytesFrame::BytesFrame(DataBuffer data_buffer, cv::Mat original_image)
    : Frame(FRAME_TYPE_BYTES, original_image), data_buffer_(data_buffer) {}

DataBuffer BytesFrame::GetDataBuffer() { return data_buffer_; }
FrameType BytesFrame::GetType() { return FRAME_TYPE_BYTES; }
