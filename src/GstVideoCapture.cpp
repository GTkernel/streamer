//
// Created by Ran Xian on 7/22/16.
//

#include "GstVideoCapture.h"

/**
 * \brief Initialize the capture with a uri. Only supports rtsp protocol.
 */
GstVideoCapture::GstVideoCapture():
  appsink_(NULL),
  pipeline_(NULL) {
}

GstVideoCapture::~GstVideoCapture() {
  if (pipeline_ != NULL) {
    DestroyPipeline();
  }
}

/**
 * \brief Destroy the pipeline, free any resources allocated.
 */
void GstVideoCapture::DestroyPipeline() {
  if (pipeline_ != NULL) {
    appsink_ = NULL;
    gst_element_set_state(GST_ELEMENT(pipeline_), GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = NULL;
  }
}

/**
 * \brief Get next frame from the pipeline.
 */
cv::Mat GstVideoCapture::GetFrame() {
  CHECK(appsink_ != NULL) << "GStreamer pipeline is not set up";
  GstSample *sample = gst_app_sink_pull_sample(appsink_);
  if (sample == NULL) {
    // EOF, return the mat
    DestroyPipeline();
    return cv::Mat();
  }
  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  gst_buffer_map(buffer, &map, GST_MAP_READ);

  cv::Mat frame(size_, CV_8UC3, (char *)map.data, cv::Mat::AUTO_STEP);

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);

  return frame;
}


cv::Size GstVideoCapture::GetFrameSize() {
  return size_;
}

/**
 * \brief Create GStreamer pipeline.
 * \param rtsp_uri The uri to rtsp endpoints.
 * \return True if the pipeline is sucessfully built.
 */
bool GstVideoCapture::CreatePipeline(std::string rtsp_uri) {
  CHECK(rtsp_uri.substr(0, 7) == "rtsp://") << "Streaming protocol other than rtsp is not supported";

  gchar *descr = g_strdup_printf(
    "rtspsrc location=\"%s\" "
    "! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! capsfilter caps=video/x-raw,format=(string)BGR "
    "! appsink name=sink sync=true",
    rtsp_uri.c_str()
  );

  GError *error = NULL;

  // Create pipeline
  LOG(INFO) << "Launch GStreamer pipeline";
  GstElement *pipeline = gst_parse_launch(descr, &error);
  g_free(descr);

  if (error != NULL) {
    LOG(INFO) << "Could not construct pipeline: " <<  error->message;
    g_error_free(error);
    return false;
  }

  // Get sink
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  gst_app_sink_set_emit_signals((GstAppSink *)sink, true);
  gst_app_sink_set_drop((GstAppSink *)sink, true);
  gst_app_sink_set_max_buffers((GstAppSink *)sink, 1);

  if (gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    LOG(FATAL) << "Could not start pipeline";
    gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    return false;
  }

  // Get caps, and other stream info
  GstSample *sample = gst_app_sink_pull_sample((GstAppSink *)sink);
  if (sample == NULL) {
    LOG(INFO) << "The video stream encounters EOS";
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    return false;
  }

  GstCaps *caps = gst_sample_get_caps(sample);
  gchar *caps_str = gst_caps_to_string(caps);
  GstStructure *structure = gst_caps_get_structure(caps, 0);
  int width, height;

  if (!gst_structure_get_int(structure, "width", &width) ||
      !gst_structure_get_int(structure, "height", &height)) {
    LOG(FATAL) << "Could not get sample dimension";
    gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    gst_sample_unref(sample);
    return false;
  }

  this->size_ = cv::Size(width, height);
  this->caps_string_ = std::string(caps_str);
  this->pipeline_ = (GstPipeline *)pipeline;
  this->appsink_ = (GstAppSink *)sink;
  g_free(caps_str);

  LOG(INFO) << "Pipeline created, video size: " << width << "x" << height;
  LOG(INFO) << "Video caps: " << caps_string_;

  gst_sample_unref(sample);

  return true;
}