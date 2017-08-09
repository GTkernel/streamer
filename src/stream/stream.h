//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#ifndef STREAMER_STREAM_STREAM_H_
#define STREAMER_STREAM_STREAM_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_set>

#include "common/common.h"
#include "common/timer.h"
#include "frame.h"

/**
 * @brief A reader that reads from a stream. There could be multiple readers
 * reading from the same stream.
 */
class StreamReader {
  friend class Stream;

 public:
  StreamReader(Stream* stream, size_t max_buffer_size = 5);

  /**
   * @brief Pop a frame, and timeout if no frame available for a given time
   * @param timeout_ms Time out threshold, 0 for forever
   */
  std::unique_ptr<Frame> PopFrame(unsigned int timeout_ms = 0);

  void UnSubscribe();
  double GetPushFps();
  double GetPopFps();
  double GetHistoricalFps();

 private:
  /**
   * @brief Push a frame into the stream.
   * @param frame The frame to be pushed into the stream.
   */
  void PushFrame(std::unique_ptr<Frame> frame);
  Stream* stream_;
  // Max size of the buffer to hold frames in the stream
  size_t max_buffer_size_;
  // The frame buffer
  std::queue<std::unique_ptr<Frame>> frame_buffer_;
  // Stream synchronization
  std::mutex mtx_;
  std::condition_variable buffer_cv_;

  // The total number of frames that have popped from this StreamReader.
  unsigned long num_frames_popped_;
  // Milliseconds between when this StreamReader was constructed and when the
  // first frame was popped. -1 means that this has not been set yet.
  double first_frame_pop_ms_;
  // Alpha parameter for the exponentially weighted moving average (EWMA)
  // formula.
  double alpha_;
  // The EWMA of the milliseconds between frame pushes.
  double running_push_ms_;
  // The EWMA of the milliseconds between frame pops.
  double running_pop_ms_;
  // Milliseconds between when this StreamReader was constructed and the last
  // frame push.
  double last_push_ms_;
  // Milliseconds between when this StreamReader was constructed and the last
  // frame pop.
  double last_pop_ms_;
  // Started when this StreamReader is constructed.
  Timer timer_;
};

/**
 * @brief A stream is a serious of data, the data itself could be stats, images,
 * or simply a bunch of bytes.
 */
class Stream {
 public:
  Stream();
  Stream(string name);
  /**
   * @brief Push a frame into the stream.
   * @param frame The frame to be pushed into the stream.
   */
  void PushFrame(std::unique_ptr<Frame> frame);

  /**
   * @brief Get the name of the stream.
   */
  string GetName() { return name_; }

  /**
   * @brief Get a reader from the stream.
   * @param max_buffer_size The buffer size limit of the reader.
   */
  StreamReader* Subscribe(size_t max_buffer_size = 5);

  /**
   * @brief Unsubscribe from the stream
   */
  void UnSubscribe(StreamReader* reader);

 private:
  // Stream name for profiling and debugging
  string name_;
  // The readers of the stream
  std::vector<std::shared_ptr<StreamReader>> readers_;
  // Stream lock
  std::mutex stream_lock_;
};

#endif  // STREAMER_STREAM_STREAM_H_
