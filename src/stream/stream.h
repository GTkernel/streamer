//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#ifndef STREAMER_STREAM_H
#define STREAMER_STREAM_H

#include "common/common.h"
#include "frame.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_set>

/**
 * @brief A reader that reads from a stream. There could be multiple readers
 * reading from the same stream.
 */
class StreamReader {
  friend class Stream;

 public:
  StreamReader(Stream *stream, size_t max_buffer_size = 5);

  /**
   * @brief Pop a frame, and timeout if no frame available for a given time
   * @param timeout_ms Time out threshold, 0 for forever
   */
  template <typename FT = Frame>
  std::shared_ptr<FT> PopFrame(unsigned int timeout_ms = 0);

  void UnSubscribe();

 private:
  /**
   * @brief Push a frame into the stream.
   * @param frame The frame to be pushed into the stream.
   */
  void PushFrame(std::shared_ptr<Frame> frame);
  // Max size of the buffer to hold frames in the stream
  size_t max_buffer_size_;
  // The frame buffer
  std::queue<std::shared_ptr<Frame>> frame_buffer_;
  // Stream synchronization
  std::mutex buffer_lock_;
  std::condition_variable buffer_cv_;
  Stream *stream_;
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
  void PushFrame(std::shared_ptr<Frame> frame);
  /**
   * @brief Push a raw pointer of the frame into the stream.
   * @param frame The frame to be pushed into the stream.
   */
  void PushFrame(Frame *frame);
  /**
   * @brief Get the name of the stream.
   */
  string GetName() { return name_; }

  /**
   * @brief Get a reader from the stream.
   * @param max_buffer_size The buffer size limit of the reader.
   */
  StreamReader *Subscribe(size_t max_buffer_size = 5);

  /**
   * @brief Unsubscribe from the stream
   */
  void UnSubscribe(StreamReader *reader);

 private:
  // Stream name for profiling and debugging
  string name_;
  // The readers of the stream
  std::vector<std::shared_ptr<StreamReader>> readers_;
  // Stream lock
  std::mutex stream_lock_;
};

#endif  // STREAMER_STREAM_H
