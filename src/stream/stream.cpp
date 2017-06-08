//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#include "stream.h"

/////// Stream
Stream::Stream() {}
Stream::Stream(string name) : name_(name) {}

StreamReader* Stream::Subscribe(size_t max_buffer_size) {
  std::lock_guard<std::mutex> guard(stream_lock_);

  std::shared_ptr<StreamReader> reader(new StreamReader(this, max_buffer_size));

  readers_.push_back(reader);

  return reader.get();
}

void Stream::UnSubscribe(StreamReader* reader) {
  std::lock_guard<std::mutex> guard(stream_lock_);

  readers_.erase(std::remove_if(readers_.begin(), readers_.end(),
                                [reader](std::shared_ptr<StreamReader> sr) {
                                  return sr.get() == reader;
                                }));
}

void Stream::PushFrame(std::shared_ptr<Frame> frame) {
  std::lock_guard<std::mutex> guard(stream_lock_);
  for (const auto& reader : readers_) {
    reader->PushFrame(frame);
  }
}

void Stream::PushFrame(Frame* frame) {
  PushFrame(std::shared_ptr<Frame>(frame));
}

/////// Stream Reader
StreamReader::StreamReader(Stream* stream, size_t max_buffer_size)
    : stream_(stream), max_buffer_size_(max_buffer_size) {}

template <typename FT>
std::shared_ptr<FT> StreamReader::PopFrame(unsigned int timeout_ms) {
  std::unique_lock<std::mutex> lk(buffer_lock_);
  if (timeout_ms > 0) {
    buffer_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                        [this] { return frame_buffer_.size() != 0; });
  } else {
    buffer_cv_.wait(lk, [this] { return frame_buffer_.size() != 0; });
  }

  if (frame_buffer_.size() != 0) {
    std::shared_ptr<Frame> frame = frame_buffer_.front();
    frame_buffer_.pop();
    return std::dynamic_pointer_cast<FT>(frame);
  } else {
    // Can't get frame within timeout
    return nullptr;
  }
}

void StreamReader::PushFrame(std::shared_ptr<Frame> frame) {
  std::lock_guard<std::mutex> lock(buffer_lock_);
  frame_buffer_.push(frame);
  while (frame_buffer_.size() > max_buffer_size_) {
    frame_buffer_.pop();
  }
  buffer_cv_.notify_all();
}

void StreamReader::UnSubscribe() { stream_->UnSubscribe(this); }

template std::shared_ptr<Frame> StreamReader::PopFrame(unsigned int);
template std::shared_ptr<ImageFrame> StreamReader::PopFrame(unsigned int);
template std::shared_ptr<MetadataFrame> StreamReader::PopFrame(unsigned int);
template std::shared_ptr<BytesFrame> StreamReader::PopFrame(unsigned int);
template std::shared_ptr<LayerFrame> StreamReader::PopFrame(unsigned int);
