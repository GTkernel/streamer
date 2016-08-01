//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_GSTVIDEOCAPTURE_H
#define TX1_DNN_GSTVIDEOCAPTURE_H

#include "common.h"
#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/gstmemory.h>
#include <mutex>

/**
 * Video capture for reading frames from GStreamer. Return frames in OpenCV BGR Mat. Internally the video capture is
 * using GStreamer-1.0's appsink to `intercept' frame buffers. When the \b GetFrame() method is called, The video
 * capture will issue a pre_preroll signal to the GStreamer pipeline.
 */
class GstVideoCapture {
public:
  GstVideoCapture();
  ~GstVideoCapture();
  cv::Mat TryGetFrame();
  cv::Mat GetFrame();
  cv::Size GetOriginalFrameSize();
  cv::Size GetTargetFrameSize();
  void SetTargetFrameSize(const cv::Size &target_size);
  bool CreatePipeline(std::string rtsp_uri);
  void DestroyPipeline();
  bool IsConnected();

private:
  static GstFlowReturn NewSampleCB(GstAppSink *appsink, gpointer data);

private:
  void CheckBuffer();
  void CheckBus();

private:
  cv::Size original_size_;
  cv::Size target_size_;
  std::string caps_string_;
  GstPipeline *pipeline_;
  GstAppSink *appsink_;
  GstBus *bus_;
  std::mutex capture_lock_;
  std::deque<cv::Mat> frames_;
  bool connected_;
};


#endif //TX1_DNN_GSTVIDEOCAPTURE_H
