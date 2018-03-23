// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Created by Ran Xian on 7/22/16.
//

#ifndef STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_
#define STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_

#include <condition_variable>
#include <mutex>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstmemory.h>
#include <opencv2/opencv.hpp>

/**
 * @brief Video capture for reading frames from GStreamer. Return frames in
 * OpenCV BGR Mat. Internally the video capture is using GStreamer-1.0's
 * appsink to `intercept' frame buffers. When the <code>GetPixels()</code>
 * method is called, The video capture will issue a <code>pre_preroll</code>
 * signal to the GStreamer pipeline.
 */
class GstVideoCapture {
 public:
  GstVideoCapture(unsigned long max_buf_size = 10);
  ~GstVideoCapture();
  cv::Mat GetPixels(unsigned long frame_id);
  cv::Size GetOriginalFrameSize() const;
  bool CreatePipeline(std::string video_uri,
                      const std::string& output_filepath = "",
                      unsigned int file_framerate = 0);
  void DestroyPipeline();
  bool IsConnected() const;

  /**
   * @brief Set Gstreamer decoder element direclty. The caller should make sure
   * that decoder element can work on the running system.
   *
   * @param decoder The name of the deocder gstreamer element.
   */
  void SetDecoderElement(const std::string& decoder);
  bool NextFrameIsLast() const;

 private:
  // Callback triggered when end of stream detected. This callback computes the
  // id of the last frame in the stream.
  static void Eos(GstAppSink* appsink, gpointer user_data);
  static GstFlowReturn NewSampleCB(GstAppSink* appsink, gpointer data);

 private:
  void CheckBuffer();
  void CheckBus();

  // This is a mapping from a GstAppSink to the GstVideoCapture object
  // associated with that GstAppSink. This map is used in the static "Eos()"
  // callback function in order to determine which GstVideoCapture object is
  // associated with the GstAppSink that triggered the callback.
  static std::unordered_map<GstAppSink*, GstVideoCapture*> appsink_to_capture_;
  // Protect access to appsink_to_capture_.
  static std::mutex mtx_;

  cv::Size original_size_;
  std::string caps_string_;
  GstAppSink* appsink_;
  GstPipeline* pipeline_;
  GstBus* bus_;
  std::mutex capture_lock_;
  // Used to apply backpressure to the GStreamer pipeline.
  std::condition_variable gst_cv_;
  std::condition_variable capture_cv_;
  std::deque<cv::Mat> frames_;
  bool connected_;
  std::string decoder_element_;
  // The monotonically-increasing id of the most recent frame to be returned by
  // "GetPixels()". This is used by the "Eos()" callback to calculate the id of
  // the last frame.
  unsigned long current_frame_id_;
  // The id of the last frame in the stream. Set by the "Eos()" callback.
  unsigned long last_frame_id_;
  // True if the end of the stream has been detected. Set by the "Eos()"
  // callback.
  bool found_last_frame_;
  unsigned long max_buf_size_;
};

#endif  // STREAMER_VIDEO_GST_VIDEO_CAPTURE_H_
