//
// Created by Ran Xian (xranthoar@gmail.com) on 9/26/16.
//

#ifndef TX1DNN_STREAM_H
#define TX1DNN_STREAM_H

#include "common/common.h"

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
  /**
   * @brief Pop a frame from the stream, the frame will be removed from the
   * stream.
   * @return The first frame in the series.
   */
  cv::Mat PopFrame();
  /**
   * @brief Push a frame into the stream.
   * @param frame The frame to be pushed into the stream.
   */
  void PushFrame(const cv::Mat &frame);

 private:
  int max_buffer_size_;
  std::queue<cv::Mat> frame_buffer_;
  std::mutex stream_lock_;
  std::condition_variable stream_cv_;
};

#endif  // TX1DNN_STREAM_H
