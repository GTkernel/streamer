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

/**
 * Video capture for reading frames from GStreamer. Return frames in OpenCV BGR Mat. Internally the video capture is
 * using GStreamer-1.0's appsink to `intercept' frame buffers. When the \b GetFrame() method is called, The video
 * capture will issue a pre_preroll signal to the GStreamer pipeline.
 */
class GstVideoCapture {
public:
  GstVideoCapture();
  ~GstVideoCapture();
  cv::Mat GetFrame();
  cv::Size GetFrameSize();
  bool CreatePipeline(std::string rtsp_uri);
  void DestroyPipeline();
private:
  cv::Size size_;
  std::string caps_string_;
  GstPipeline *pipeline_;
  GstAppSink *appsink_;
};


#endif //TX1_DNN_GSTVIDEOCAPTURE_H
