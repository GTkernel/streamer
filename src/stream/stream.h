//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#ifndef TX1DNN_STREAM_H
#define TX1DNN_STREAM_H

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
   * @brief Pop a frame from the stream, the frame will be removed from the
   * stream.
   * @return The first frame in the series.
   */
  std::shared_ptr<Frame> PopFrame();
  std::shared_ptr<ImageFrame> PopImageFrame();
  std::shared_ptr<MetadataFrame> PopMDFrame();
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

#endif  // TX1DNN_STREAM_H
