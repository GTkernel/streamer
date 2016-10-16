//
// Created by Ran Xian (xranthoar@gmail.com) on 10/13/16.
//

#ifndef TX1DNN_GST_VIDEO_ENCODER_H
#define TX1DNN_GST_VIDEO_ENCODER_H

#include "common/common.h"
#include "gst/app/gstappsrc.h"
#include "gst/gst.h"
#include "gst/gstbuffer.h"
#include "processor/processor.h"

#include <mutex>
/**
 * @brief Video encoder utilizing GStreamer pipeline
 */
class GstVideoEncoder : public Processor {
 public:
  GstVideoEncoder(StreamPtr stream, int width, int height,
                  const string &output_filename);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string BuildPipelineString();
  string BuildCapsString();

  // GST Callbacks
  static void NeedDataCB(GstAppSrc *appsrc, guint size, gpointer user_data);
  static void EnoughDataCB(GstAppSrc *appsrc, gpointer userdata);

  // Video attributes
  int width_;
  int height_;
  int frame_size_bytes_;
  string output_filename_;

  // Gst elements
  GstBus *gst_bus_;
  GstPipeline *gst_pipeline_;
  GstAppSrc *gst_appsrc_;
  GstCaps *gst_caps_;
  GMainLoop *g_main_loop_;

  // States
  bool need_data_;

  // Lock
  std::mutex encoder_lock_;
};

#endif  // TX1DNN_GST_VIDEO_ENCODER_H