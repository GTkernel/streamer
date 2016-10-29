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

/**
 * @brief A stream is a serious of data, the data itself could be stats, images,
 * or simply a bunch of bytes.
 */
class Stream {
 public:
  Stream(int max_buffer_size = 5);
  Stream(string name, int max_buffer_size = 5);
  /**
   * @brief Pop a frame of type FT from the stream.
   * @return  The head frame in the stream.
   */
  template <typename FT = Frame>
  std::shared_ptr<FT> PopFrame();
  /**
   * @brief Push a frame into the stream.
   * @param frwame The frame to be pushed into the stream.
   */
  void PushFrame(std::shared_ptr<Frame> frame);
  void PushFrame(Frame *frame);
  void SetName(const string &name) { name_ = name; }
  string GetName() { return name_; }

 private:
  int max_buffer_size_;
  std::queue<std::shared_ptr<Frame>> frame_buffer_;
  std::mutex stream_lock_;
  std::condition_variable stream_cv_;
  // For profiling and debugging
  string name_;
};

#endif  // STREAMER_STREAM_H
