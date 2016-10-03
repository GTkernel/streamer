//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_GSTVIDEOCAPTURE_H
#define TX1_DNN_GSTVIDEOCAPTURE_H

#include "common/common.h"
#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/gstmemory.h>
#include <mutex>
#include <condition_variable>

/**
 * @brief Video capture for reading frames from GStreamer. Return frames in
 * OpenCV BGR Mat. Internally the video capture is using GStreamer-1.0's
 * appsink to `intercept' frame buffers. When the <code>GetFrame()</code>
 * method is called, The video capture will issue a <code>pre_preroll</code>
 * signal to the GStreamer pipeline.
 */
class GstVideoCapture {
 public:
  GstVideoCapture();
  ~GstVideoCapture();
  cv::Mat TryGetFrame(DataBuffer *data_bufferp = nullptr);
  cv::Mat GetFrame(DataBuffer *data_bufferp = nullptr);
  cv::Size GetOriginalFrameSize();
  bool CreatePipeline(std::string video_uri);
  void DestroyPipeline();
  bool IsConnected();

 private:
  static GstFlowReturn NewSampleCB(GstAppSink *appsink, gpointer data);

 private:
  void CheckBuffer();
  void CheckBus();

 private:
  cv::Size original_size_;
  std::string caps_string_;
  GstPipeline *pipeline_;
  GstAppSink *appsink_;
  GstBus *bus_;
  std::mutex capture_lock_;
  std::condition_variable capture_cv_;
  std::deque<cv::Mat> frames_;
  bool connected_;
};

#endif //TX1_DNN_GSTVIDEOCAPTURE_H
