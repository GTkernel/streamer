//
// Created by Ran Xian on 7/22/16.
//

#ifndef STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_
#define STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstmemory.h>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "common/common.h"

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
  cv::Mat TryGetFrame();
  cv::Mat GetFrame();
  cv::Size GetOriginalFrameSize() const;
  bool CreatePipeline(std::string video_uri);
  void DestroyPipeline();
  bool IsConnected() const;

  /**
   * @brief Set Gstreamer decoder element direclty. The caller should make sure
   * that decoder element can work on the running system.
   *
   * @param decoder The name of the deocder gstreamer element.
   */
  void SetDecoderElement(const string& decoder);

 private:
  static GstFlowReturn NewSampleCB(GstAppSink* appsink, gpointer data);

 private:
  void CheckBuffer();
  void CheckBus();

 private:
  cv::Size original_size_;
  std::string caps_string_;
  GstAppSink* appsink_;
  GstPipeline* pipeline_;
  GstBus* bus_;
  std::mutex capture_lock_;
  std::condition_variable capture_cv_;
  std::deque<cv::Mat> frames_;
  bool connected_;

  string decoder_element_;
};

#endif  // STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_
