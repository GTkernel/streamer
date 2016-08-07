//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_GSTVIDEOCAPTURE_H
#define TX1_DNN_GSTVIDEOCAPTURE_H

#include "common.h"
#include "classifier.h"
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
  cv::Mat TryGetFrame(DataBuffer *data_bufferp = nullptr);
  cv::Mat GetFrame(DataBuffer *data_bufferp = nullptr);
  cv::Size GetOriginalFrameSize();
  // TODO: Consider extract preprocessor logic to a configurable data member for better readability.
  bool CreatePipeline(std::string rtsp_uri);
  void DestroyPipeline();
  bool IsConnected();
  void SetPreprocessClassifier(std::shared_ptr<Classifier> classifier);

private:
  static GstFlowReturn NewSampleCB(GstAppSink *appsink, gpointer data);

private:
  void CheckBuffer();
  void CheckBus();
  bool IsPreprocessed();

private:
  cv::Size original_size_;
  std::string caps_string_;
  GstPipeline *pipeline_;
  GstAppSink *appsink_;
  GstBus *bus_;
  std::mutex capture_lock_;
  std::deque<cv::Mat> frames_;
  std::deque<DataBuffer> preprocessed_buffers_;
  bool connected_;
  std::shared_ptr<Classifier> preprocess_classifier_;
};


#endif //TX1_DNN_GSTVIDEOCAPTURE_H
