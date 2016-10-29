//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#include "stream.h"

Stream::Stream(int max_buffer_size) : max_buffer_size_(max_buffer_size) {}

Stream::Stream(string name, int max_buffer_size)
    : name_(name), max_buffer_size_(max_buffer_size) {}

template <typename FT>
std::shared_ptr<FT> Stream::PopFrame() {
  Timer timer;
  timer.Start();
  std::unique_lock<std::mutex> lk(stream_lock_);
  stream_cv_.wait(lk, [this] { return frame_buffer_.size() != 0; });
  std::shared_ptr<Frame> frame = frame_buffer_.front();
  frame_buffer_.pop();

  return std::dynamic_pointer_cast<FT>(frame);
}

void Stream::PushFrame(std::shared_ptr<Frame> frame) {
  std::lock_guard<std::mutex> lock(stream_lock_);
  frame_buffer_.push(frame);
  while (frame_buffer_.size() > max_buffer_size_) {
    frame_buffer_.pop();
  }
  stream_cv_.notify_all();
}

void Stream::PushFrame(Frame *frame) {
  PushFrame(std::shared_ptr<Frame>(frame));
}

template std::shared_ptr<Frame> Stream::PopFrame();
template std::shared_ptr<ImageFrame> Stream::PopFrame();
template std::shared_ptr<MetadataFrame> Stream::PopFrame();
template std::shared_ptr<BytesFrame> Stream::PopFrame();
