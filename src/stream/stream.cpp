//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#include "stream/stream.h"

#include "processor/flow_control/flow_control_entrance.h"

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

void Stream::PushFrame(std::unique_ptr<Frame> frame) {
  std::lock_guard<std::mutex> guard(stream_lock_);
  if (readers_.size() == 1) {
    readers_.at(0)->PushFrame(std::move(frame));
  } else {
    // If there is more than one reader, then we need to copy the frame.
    for (const auto& reader : readers_) {
      reader->PushFrame(std::make_unique<Frame>(frame));
    }
  }
}

/////// StreamReader
StreamReader::StreamReader(Stream* stream, size_t max_buffer_size)
    : stream_(stream),
      max_buffer_size_(max_buffer_size),
      num_frames_popped_(0),
      first_frame_pop_ms_(-1),
      alpha_(0.25),
      running_push_ms_(0),
      running_pop_ms_(0),
      last_push_ms_(0),
      last_pop_ms_(0) {
  timer_.Start();
}

std::unique_ptr<Frame> StreamReader::PopFrame(unsigned int timeout_ms) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (timeout_ms > 0) {
    buffer_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this] { return frame_buffer_.size() != 0; });
  } else {
    buffer_cv_.wait(lock, [this] { return frame_buffer_.size() != 0; });
  }

  if (frame_buffer_.size() != 0) {
    auto frame = std::move(frame_buffer_.front());
    frame_buffer_.pop();
    ++num_frames_popped_;

    double current_ms = timer_.ElapsedMSec();
    double delta_ms = current_ms - last_pop_ms_;
    running_pop_ms_ = running_pop_ms_ * (1 - alpha_) + delta_ms * alpha_;
    last_pop_ms_ = current_ms;

    if (first_frame_pop_ms_ == -1) {
      first_frame_pop_ms_ = current_ms;
    }

    return frame;
  } else {
    // Can't get frame within timeout
    return nullptr;
  }
}

void StreamReader::PushFrame(std::unique_ptr<Frame> frame) {
  std::lock_guard<std::mutex> lock(mtx_);

  double current_ms = timer_.ElapsedMSec();
  double delta_ms = current_ms - last_push_ms_;
  running_push_ms_ = running_push_ms_ * (1 - alpha_) + delta_ms * alpha_;
  last_push_ms_ = current_ms;

  // If buffer is full, the frame is dropped
  if (frame_buffer_.size() < max_buffer_size_) {
    frame_buffer_.push(std::move(frame));
  } else {
    unsigned long id = frame->GetValue<unsigned long>("frame_id");
    LOG(WARNING) << "Stream queue full. Dropping frame: " << id;
    if (frame->GetFlowControlEntrance()) {
      // This scenario should not happen. If we're using end-to-end flow
      // control, then we should not be using so many tokens such that we are
      // dropping frames.
      LOG(ERROR) << "Dropped frame " << id << " while using end-to-end flow "
                 << "control. This should not have happened. Either increase "
                 << "the size of this stream's queue or decrease the number of "
                 << "flow control tokens.";
    }
  }
  buffer_cv_.notify_all();
}

void StreamReader::UnSubscribe() { stream_->UnSubscribe(this); }

double StreamReader::GetPushFps() { return 1000 / running_push_ms_; }

double StreamReader::GetPopFps() { return 1000 / running_pop_ms_; }

double StreamReader::GetHistoricalFps() {
  return num_frames_popped_ /
         ((timer_.ElapsedMSec() - first_frame_pop_ms_) / 1000);
}
