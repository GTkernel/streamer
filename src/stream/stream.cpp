//
// Created by xianran on 9/26/16.
//

#include "stream.h"

Stream::Stream(int max_buffer_size) : max_buffer_size_(max_buffer_size) {}

cv::Mat Stream::PopFrame() {
  std::unique_lock<std::mutex> lk(stream_lock_);
  stream_cv_.wait(lk, [this] {
    return frame_buffer_.size() != 0;
  });
  cv::Mat frame = frame_buffer_.front();
  frame_buffer_.pop();

  return frame;
}

void Stream::PushFrame(const cv::Mat &frame) {
  std::lock_guard<std::mutex> lock(stream_lock_);
  frame_buffer_.push(frame);
  while (frame_buffer_.size() > max_buffer_size_) {
    frame_buffer_.pop();
  }
  stream_cv_.notify_all();
}
