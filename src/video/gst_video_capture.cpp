//
// Created by Ran Xian on 7/22/16.
//

#include "video/gst_video_capture.h"

#include <chrono>
#include <thread>

#include <gst/app/gstappsink.h>
#include <gst/gstmemory.h>

#include "common/context.h"
#include "utils/utils.h"

std::unordered_map<GstAppSink*, GstVideoCapture*>
    GstVideoCapture::appsink_to_capture_;
std::mutex GstVideoCapture::mtx_;

/************************
 * GStreamer callbacks ***
 ************************/
void GstVideoCapture::Eos(GstAppSink* appsink, gpointer) {
  GstVideoCapture* cap;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    cap = appsink_to_capture_.at(appsink);
  }
  cap->found_last_frame_ = true;
  // The id of the last frame to be received from the camera is equal to the id
  // of the most recent frame to be given to the user plus the length of the
  // frame buffer.
  cap->last_frame_id_ = cap->current_frame_id_ + cap->frames_.size();
}

/**
 * @brief Callback when there is new sample on the pipleline. First check the
 * buffer for new samples, then check the
 * bus for new message.
 * @param appsink App sink that has has available sample.
 * @param data User defined data pointer.
 * @return Always return OK as we want to continuously listen for new samples.
 */
GstFlowReturn GstVideoCapture::NewSampleCB(GstAppSink*, gpointer data) {
  CHECK(data != NULL) << "Callback is not passed in a capture";
  GstVideoCapture* capture = (GstVideoCapture*)data;
  capture->CheckBuffer();
  capture->CheckBus();

  return GST_FLOW_OK;
}

/**
 * @brief Pull the buffer to get frames.
 */
void GstVideoCapture::CheckBuffer() {
  if (!connected_) {
    LOG(INFO) << "Not connected";
    return;
  }

  GstSample* sample = gst_app_sink_pull_sample(appsink_);

  if (sample == NULL) {
    // No sample pulled
    LOG(INFO) << "GStreamer pulls null data, ignoring";
    return;
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
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

  CHECK_NE(original_size_.area(), 0)
      << "Capture should have got frame size information but not";
  cv::Mat frame_(original_size_, CV_8UC3, (char*)map.data, cv::Mat::AUTO_STEP);
  // Copy the data out
  cv::Mat frame = frame_.clone();

  CHECK(frame.size[1] == original_size_.width);
  CHECK(frame.size[0] == original_size_.height);

  // Push the frame
  {
    std::lock_guard<std::mutex> guard(capture_lock_);
    frames_.push_back(frame);
    if (frames_.size() > 1) {
      LOG(ERROR) << "Pipeline failed to consume frame in time, "
                 << " gst_video_capture internal frames queue size: "
                 << frames_.size();
    }
  }
  capture_cv_.notify_all();

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
}

/**
 * @brief Check the buffer to get new messages.
 */
void GstVideoCapture::CheckBus() {
  while (true) {
    GstMessage* msg = gst_bus_pop(bus_);
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
 * @brief Initialize the capture with a uri. Only supports rtsp protocol now.
 */
GstVideoCapture::GstVideoCapture()
    : appsink_(nullptr),
      pipeline_(nullptr),
      connected_(false),
      current_frame_id_(0),
      last_frame_id_(0),
      found_last_frame_(false) {
  // Get decoder element
  decoder_element_ = Context::GetContext().GetString(H264_DECODER_GST_ELEMENT);
}

GstVideoCapture::~GstVideoCapture() {
  if (connected_) {
    DestroyPipeline();
  }
}

/**
 * @brief Check if the video capture is connected to the pipeline. If the
 * video capture is not connected, app should not pull from the capture anymore.
 * @return True if connected. False otherwise.
 */
bool GstVideoCapture::IsConnected() const { return connected_; }

/**
 * @brief Destroy the pipeline, free any resources allocated.
 */
void GstVideoCapture::DestroyPipeline() {
  std::lock_guard<std::mutex> guard(capture_lock_);
  if (!connected_) return;

  appsink_ = NULL;
  if (gst_element_set_state(GST_ELEMENT(pipeline_), GST_STATE_NULL) !=
      GST_STATE_CHANGE_SUCCESS) {
    LOG(ERROR) << "Can't set pipeline state to NULL";
  }
  gst_object_unref(pipeline_);
  pipeline_ = NULL;

  connected_ = false;
}

/**
 * @brief Get next frame from the pipeline, busy wait (which should be improved)
 * until frame available.
 */
cv::Mat GstVideoCapture::GetPixels(unsigned long frame_id) {
  current_frame_id_ = frame_id;

  Timer timer;
  timer.Start();
  if (!connected_) return cv::Mat();

  std::unique_lock<std::mutex> lk(capture_lock_);
  bool pred = capture_cv_.wait_for(lk, std::chrono::milliseconds(100), [this] {
    // Stop waiting when frame available or connection fails
    return !connected_ || frames_.size() != 0;
  });

  if (!pred) {
    // The wait stopped because of a timeout.
    return cv::Mat();
  }
  if (!connected_) return cv::Mat();

  cv::Mat pixels = frames_.front();
  frames_.pop_front();

  return pixels;
}

/**
 * @brief Get the size of original frame.
 */
cv::Size GstVideoCapture::GetOriginalFrameSize() const {
  return original_size_;
}

/**
 * @brief Create GStreamer pipeline.
 * @param video_uri The uri to video source, could be rtsp endpoint or facetime.
 * If it is facetime, will try to use macbook's facetime camera.
 * @return True if the pipeline is sucessfully built.
 */
bool GstVideoCapture::CreatePipeline(std::string video_uri,
                                     const std::string& output_filepath,
                                     unsigned int file_framerate) {
  // The pipeline that emits video frames
  std::ostringstream pipeline;

  std::string video_protocol;
  std::string video_path;
  ParseProtocolAndPath(video_uri, video_protocol, video_path);

  if (video_protocol == "rtsp") {
    pipeline << "rtspsrc latency=0 location=\"" << video_uri << "\""
             << " ! rtph264depay ! h264parse ! ";
    if (output_filepath != "") {
      pipeline << "tee name=t ! queue ! ";
    }
    pipeline << decoder_element_;
  } else if (video_protocol == "gst") {
    LOG(WARNING) << "Directly use gst pipeline as video pipeline";
    pipeline << video_path;
    LOG(INFO) << pipeline.str();
  } else if (video_protocol == "file") {
    LOG(WARNING) << "Reading H.264-encoded data from file using GStreamer";
    pipeline << "filesrc location=\"" << video_path
             << "\" ! qtdemux ! h264parse ! ";
    if (output_filepath != "") {
      pipeline << "tee name=t ! queue ! ";
    }
    pipeline << decoder_element_;
    if (file_framerate > 0) {
      pipeline << " ! videorate ! video/x-raw,framerate=" << file_framerate
               << "/1";
    }
  } else {
    LOG(FATAL) << "Video uri: " << video_uri << " is not valid";
  }

  pipeline << " ! videoconvert "
              "! capsfilter caps=video/x-raw,format=(string)BGR "
              "! appsink name=sink sync=true";
  if (output_filepath != "" &&
      (video_protocol == "rtsp" || video_protocol == "file")) {
    pipeline << " t. ! queue ! mp4mux ! filesink location=" << output_filepath;
  }

  gchar* descr = g_strdup_printf("%s", pipeline.str().c_str());
  LOG(INFO) << "Capture video pipeline: " << descr;

  GError* error = NULL;

  // Create pipeline
  GstElement* gst_pipeline = gst_parse_launch(descr, &error);
  LOG(INFO) << "GStreamer pipeline launched";
  g_free(descr);

  if (error != NULL) {
    LOG(ERROR) << "Could not construct pipeline: " << error->message;
    g_error_free(error);
    return false;
  }

  // Get sink
  GstAppSink* sink =
      GST_APP_SINK(gst_bin_get_by_name(GST_BIN(gst_pipeline), "sink"));
  {
    std::lock_guard<std::mutex> lock(mtx_);
    appsink_to_capture_[sink] = this;
  }
  gst_app_sink_set_emit_signals(sink, true);
  gst_app_sink_set_drop(sink, true);
  gst_app_sink_set_max_buffers(sink, 1);

  // Get bus
  GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(gst_pipeline));
  if (bus == NULL) {
    LOG(ERROR) << "Can't get bus from pipeline";
    return false;
  }
  this->bus_ = bus;

  // Get stream info
  if (gst_element_set_state(gst_pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not start pipeline";
    gst_element_set_state(gst_pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(gst_pipeline));
    return false;
  }

  // Get caps, and other stream info
  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (sample == NULL) {
    LOG(INFO) << "The video stream encounters EOS";
    gst_element_set_state(gst_pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(gst_pipeline));
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  gst_sample_unref(sample);
  gchar* caps_str = gst_caps_to_string(caps);
  GstStructure* structure = gst_caps_get_structure(caps, 0);
  int width, height;

  if (!gst_structure_get_int(structure, "width", &width) ||
      !gst_structure_get_int(structure, "height", &height)) {
    LOG(ERROR) << "Could not get sample dimension";
    gst_element_set_state(gst_pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(gst_pipeline));
    gst_sample_unref(sample);
    return false;
  }

  this->original_size_ = cv::Size(width, height);
  this->caps_string_ = std::string(caps_str);
  g_free(caps_str);
  this->pipeline_ = (GstPipeline*)gst_pipeline;
  this->appsink_ = sink;
  this->connected_ = true;

  // Set callbacks
  if (gst_element_change_state(gst_pipeline,
                               GST_STATE_CHANGE_PLAYING_TO_PAUSED) ==
      GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not pause pipeline";
    DestroyPipeline();
    return false;
  }

  GstAppSinkCallbacks callbacks;
  callbacks.eos = GstVideoCapture::Eos;
  callbacks.new_preroll = NULL;
  callbacks.new_sample = GstVideoCapture::NewSampleCB;
  gst_app_sink_set_callbacks(sink, &callbacks, (void*)this, NULL);

  if (gst_element_set_state(gst_pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    LOG(ERROR) << "Could not start pipeline";
    DestroyPipeline();
    return false;
  }

  CheckBus();

  LOG(INFO) << "Pipeline connected, video size: " << width << "x" << height;
  LOG(INFO) << "Video caps: " << caps_string_;

  return true;
}

void GstVideoCapture::SetDecoderElement(const string& decoder) {
  decoder_element_ = decoder;
}

bool GstVideoCapture::NextFrameIsLast() const {
  return found_last_frame_ && (current_frame_id_ == last_frame_id_);
}
