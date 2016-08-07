//
// Created by Ran Xian on 7/22/16.
//

#include <gst/app/gstappsink.h>
#include <thread>
#include "gst_video_capture.h"
/************************
* GStreamer callbacks ***
************************/
/**
 * \brief Callback when there is new sample on the pipleline. First check the buffer for new samples, then check the
 * bus for new message.
 * \param appsink App sink that has has available sample.
 * \param data User defined data pointer.
 * \return Always return OK as we want to continuously listen for new samples.
 */
GstFlowReturn GstVideoCapture::NewSampleCB(GstAppSink *appsink, gpointer data) {
  CHECK(data != NULL) << "Callback is not passed in a capture";
  GstVideoCapture *capture = (GstVideoCapture *)data;
  capture->CheckBuffer();
  capture->CheckBus();

  return GST_FLOW_OK;
}

/**
 * \brief Pull the buffer to get frames.
 */
void GstVideoCapture::CheckBuffer() {
  if (!connected_) {
    LOG(INFO) << "Not connected";
    return;
  }

  CHECK(connected_) << "Capture is not connected yet";
  GstSample *sample = gst_app_sink_pull_sample(appsink_);

  if (sample == NULL) {
    // No sample pulled
    LOG(INFO) << "GStreamer pulls null data, ignoring";
    return;
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  if (buffer == NULL) {
    // No buffer
    LOG(INFO) << "GST sample has NULL buffer, ignoring";
    gst_sample_unref(sample);
    return;
  }

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    LOG(INFO) << "Can't map GST buffer to map, ignoring";
    gst_sample_unref(sample);
    return;
  }

  CHECK_NE(original_size_.area(), 0) << "Capture should have got frame size information but not";
  cv::Mat frame(original_size_, CV_8UC3, (char *)map.data, cv::Mat::AUTO_STEP);
  cv::Mat resized_frame;
  // Do preprocessing if a classifier is specified;
  if (IsPreprocessed()) {
    DataBuffer data_buffer(preprocess_classifier_->GetInputBufferSize());
    preprocess_classifier_->Preprocess(frame, data_buffer);
    preprocessed_buffers_.push_back(data_buffer);
  }

  // Push the frame
  {
    std::lock_guard<std::mutex> guard(capture_lock_);
    frames_.clear();
    frames_.push_back(frame);
  }

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
}

/**
 * \brief Check the buffer to get new messages.
 */
void GstVideoCapture::CheckBus() {
  while (true) {
    GstMessage *msg = gst_bus_pop(bus_);
    if (msg == NULL) {
      break;
    }
    gst_message_unref(msg);
  }
}

/**********************
 * GstVideoCapture code
 **********************/
/**
 * \brief Initialize the capture with a uri. Only supports rtsp protocol now.
 */
GstVideoCapture::GstVideoCapture():
  appsink_(nullptr),
  pipeline_(nullptr),
  connected_(false),
  preprocess_classifier_(nullptr) {
}

GstVideoCapture::~GstVideoCapture() {
  if (connected_) {
    DestroyPipeline();
  }
}

/**
 * \brief Check if the video capture is connected to the pipeline. If the video capture is not connected, app should not
 * pull from the capture anymore.
 * \return True if connected. False otherwise.
 */
bool GstVideoCapture::IsConnected() {
  return connected_;
}

/**
 * \brief Destroy the pipeline, free any resources allocated.
 */
void GstVideoCapture::DestroyPipeline() {
  std::lock_guard<std::mutex> guard(this->capture_lock_);
  if (!connected_)
    return;

  appsink_ = NULL;
  if (gst_element_set_state(GST_ELEMENT(pipeline_), GST_STATE_NULL) != GST_STATE_CHANGE_SUCCESS) {
    LOG(ERROR) << "Can't set pipeline state to NULL";
  }
  gst_object_unref(pipeline_);
  pipeline_ = NULL;

  connected_ = false;
}

/**
 * \brief Get next frame from the pipeline, busy wait (which should be improved) until frame available.
 */
cv::Mat GstVideoCapture::GetFrame(DataBuffer *data_bufferp) {
  Timer timer; timer.Start();
  if (!connected_)
    return cv::Mat();

  while (connected_ && frames_.size() == 0)
    ;

  LOG(INFO) << "Waited " << timer.ElapsedMSec() << " until frame available";

  if (!connected_)
    return cv::Mat();

  // There must be a frame now
  return TryGetFrame(data_bufferp);
}

/**
 * \brief Try to get next frame from the pipeline and does not block.
 * \return The frame currently in the frames queue. An empty Mat if no frame immediately available.
 */
cv::Mat GstVideoCapture::TryGetFrame(DataBuffer *data_bufferp) {
  Timer timer; timer.Start();
  if (!connected_ || frames_.size() == 0) {
    return cv::Mat();
  } else {
    std::lock_guard<std::mutex> guard(capture_lock_);
    cv::Mat frame = frames_.front();
    frames_.pop_front();

    if (IsPreprocessed()) {
      DataBuffer preprocessed_buffer = preprocessed_buffers_.front();;
      preprocessed_buffers_.pop_front();
      if (data_bufferp != nullptr) {
        CHECK(preprocess_classifier_ != nullptr) << "Can't get preprocessed buffer as no classifer is specified";
        *data_bufferp = preprocessed_buffer;
      }
    }

    LOG(INFO) << "Get frame in " << timer.ElapsedMSec() << " ms";
    return frame;
  }
}

/**
 * \brief Get the size of original frame.
 */
cv::Size GstVideoCapture::GetOriginalFrameSize() {
  return original_size_;
}

/**
 * @brief Set preprocess classifier, this is an optimization to offload the preprocessing step to the video pipeline.
 * 
 * @param classifier The classifier that needs this preprocessing. The classifier's Preprocess() will be called.
 */
void GstVideoCapture::SetPreprocessClassifier(std::shared_ptr<Classifier> classifier) {
  CHECK(!connected_) << "Pipeline has already connected, can't set preprocess classifier";
  preprocess_classifier_ = classifier;
}

bool GstVideoCapture::IsPreprocessed() {
  return preprocess_classifier_ != nullptr;
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
  GstElement *pipeline = gst_parse_launch(descr, &error);
  LOG(INFO) << "GStreamer pipeline launched";
  g_free(descr);

  if (error != NULL) {
    LOG(ERROR) << "Could not construct pipeline: " <<  error->message;
    g_error_free(error);
    return false;
  }

  // Get sink
  GstAppSink *sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline), "sink"));
  gst_app_sink_set_emit_signals(sink, true);
  gst_app_sink_set_drop(sink, true);
  gst_app_sink_set_max_buffers(sink, 1);

  // Get bus
  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  if (bus == NULL) {
    LOG(ERROR) << "Can't get bus from pipeline";
    return false;
  }
  this->bus_ = bus;

  // Get stream info
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not start pipeline";
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    return false;
  }

  // Get caps, and other stream info
  GstSample *sample = gst_app_sink_pull_sample(sink);
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
    LOG(ERROR) << "Could not get sample dimension";
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    gst_sample_unref(sample);
    return false;
  }

  this->original_size_ = cv::Size(width, height);
  this->caps_string_ = std::string(caps_str);
  this->pipeline_ = (GstPipeline *)pipeline;
  this->appsink_ = sink;
  this->connected_ = true;
  g_free(caps_str);
  gst_sample_unref(sample);

  // Set callbacks
  if (gst_element_change_state(pipeline, GST_STATE_CHANGE_PLAYING_TO_PAUSED) == GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not pause pipeline";
    DestroyPipeline();
    return false;
  }

  GstAppSinkCallbacks callbacks;
  callbacks.eos = NULL;
  callbacks.new_preroll = NULL;
  callbacks.new_sample = GstVideoCapture::NewSampleCB;
  gst_app_sink_set_callbacks(sink, &callbacks, (void *)this, NULL);

  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not start pipeline";
    DestroyPipeline();
    return false;
  }

  CheckBus();

  LOG(INFO) << "Pipeline connected, video size: " << width << "x" << height;
  LOG(INFO) << "Video caps: " << caps_string_;

  return true;
}