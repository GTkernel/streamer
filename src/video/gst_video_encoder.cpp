//
// Created by Ran Xian (xranthoar@gmail.com) on 10/13/16.
//

#include "gst_video_encoder.h"
#include "common/context.h"

const static char *ENCODER_SRC_NAME = "encoder_src";

GstVideoEncoder::GstVideoEncoder(StreamPtr input_stream, int width, int height,
                                 const string &output_filename)
    : width_(width),
      height_(height),
      frame_size_bytes_(width * height * 3),
      output_filename_(output_filename),
      need_data_(false),
      timestamp_(0) {
  sources_.push_back(input_stream);
  // Encoder
  encoder_element_ = Context::GetContext().GetString(H264_ENCODER_GST_ELEMENT);
}

void GstVideoEncoder::NeedDataCB(GstAppSrc *appsrc, guint size,
                                 gpointer user_data) {
  if (user_data == nullptr) return;

  GstVideoEncoder *encoder = (GstVideoEncoder *)user_data;
  if (encoder->IsStarted()) encoder->need_data_ = true;
}

void GstVideoEncoder::EnoughDataCB(GstAppSrc *appsrc, gpointer user_data) {
  DLOG(INFO) << "Received enough data signal";
  if (user_data == nullptr) return;

  GstVideoEncoder *encoder = (GstVideoEncoder *)user_data;
  encoder->need_data_ = false;
}

string GstVideoEncoder::BuildPipelineString() {
  std::ostringstream ss;
  ss << "appsrc name=" << ENCODER_SRC_NAME << " ! "
     << "videoconvert ! " << encoder_element_ << " ! "
     << "qtmux ! filesink location=" << output_filename_;

  DLOG(INFO) << "Encoder pipeline is " << ss.str();

  return ss.str();
}

string GstVideoEncoder::BuildCapsString() {
  std::ostringstream ss;
  ss << "video/x-raw,format=(string)BGR,width=" << width_
     << ",height=" << height_ << ",framerate=(fraction)30/1";

  DLOG(INFO) << "Encoder caps string is " << ss.str();

  return ss.str();
}

bool GstVideoEncoder::Init() {
  if (width_ == 0 || height_ == 0) {
    LOG(ERROR) << "width or height of output video is not valid (" << width_
               << "x" << height_ << ")";
    return false;
  }

  g_main_loop_ = g_main_loop_new(NULL, FALSE);

  // Build Gst pipeline
  GError *err = nullptr;
  string pipeline_str = BuildPipelineString();

  gst_pipeline_ = GST_PIPELINE(gst_parse_launch(pipeline_str.c_str(), &err));
  if (err != nullptr) {
    LOG(ERROR) << "gstreamer failed to launch pipeline: " << pipeline_str;
    LOG(ERROR) << err->message;
    g_error_free(err);
    return false;
  }

  if (gst_pipeline_ == nullptr) {
    LOG(ERROR) << "Failed to convert gst_element to gst_pipeline";
    return false;
  }

  gst_bus_ = gst_pipeline_get_bus(gst_pipeline_);
  if (gst_bus_ == nullptr) {
    LOG(ERROR) << "Failed to retrieve gst_bus from gst_pipeline";
    return false;
  }

  // Get the appsrc and connect callbacks

  GstElement *appsrc_element =
      gst_bin_get_by_name(GST_BIN(gst_pipeline_), ENCODER_SRC_NAME);
  GstAppSrc *appsrc = GST_APP_SRC(appsrc_element);

  if (appsrc == nullptr) {
    LOG(ERROR) << "Failed to get appsrc from pipeline";
    return false;
  }

  gst_appsrc_ = appsrc;

  // Set the caps of the appsrc
  string caps_str = BuildCapsString();
  gst_caps_ = gst_caps_from_string(caps_str.c_str());

  if (gst_caps_ == nullptr) {
    LOG(ERROR) << "Failed to parse caps from caps string";
    return false;
  }

  gst_app_src_set_caps(gst_appsrc_,
                       gst_caps_from_string(BuildCapsString().c_str()));
  g_object_set(G_OBJECT(gst_appsrc_), "stream-type", 0, "format",
               GST_FORMAT_TIME, NULL);

  GstAppSrcCallbacks callbacks;
  callbacks.enough_data = GstVideoEncoder::EnoughDataCB;
  callbacks.need_data = GstVideoEncoder::NeedDataCB;
  callbacks.seek_data = NULL;
  gst_app_src_set_callbacks(gst_appsrc_, &callbacks, (void *)this, NULL);

  // Play the pipeline
  GstStateChangeReturn result =
      gst_element_set_state(GST_ELEMENT(gst_pipeline_), GST_STATE_PLAYING);

  if (result != GST_STATE_CHANGE_ASYNC && result != GST_STATE_CHANGE_SUCCESS) {
    LOG(ERROR) << "Can't start gst pipeline";
    return false;
  }

  LOG(INFO) << "Pipeline launched";

  return true;
}

bool GstVideoEncoder::OnStop() {
  // Lock the state of the encoder
  std::lock_guard<std::mutex> guard(encoder_lock_);

  need_data_ = false;
  LOG(INFO) << "Stopping gstreamer pipeline";
  gst_app_src_end_of_stream(gst_appsrc_);

  const int WAIT_UNTIL_EOS_SENT_MS = 200;
  std::this_thread::sleep_for(
      std::chrono::milliseconds(WAIT_UNTIL_EOS_SENT_MS));

  GstStateChangeReturn ret =
      gst_element_set_state(GST_ELEMENT(gst_pipeline_), GST_STATE_NULL);

  if (ret != GST_STATE_CHANGE_SUCCESS) {
    LOG(ERROR) << "GStreamer failed to stop the pipeline";
  }

  LOG(INFO) << "Pipeline stopped";

  return true;
}

void GstVideoEncoder::Process() {
  auto input_frame = sources_[0]->PopImageFrame();

  // Lock the state of the encoder
  std::lock_guard<std::mutex> guard(encoder_lock_);

  if (!need_data_) return;

  // Give PTS to the buffer
  GstBuffer *buffer = gst_buffer_new_wrapped(
      input_frame->GetOriginalImage().data, frame_size_bytes_);
  GST_BUFFER_PTS(buffer) = timestamp_;
  GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, 30);
  timestamp_ += GST_BUFFER_DURATION(buffer);

  GstFlowReturn ret;
  g_signal_emit_by_name(gst_appsrc_, "push-buffer", buffer, &ret);

  if (ret != 0) {
    LOG(INFO) << "Encoder -- appsrc can't push buffer, ret code (" << ret
              << ")";
  }

  // Poll messages from the bus
  while (true) {
    GstMessage *msg = gst_bus_pop(gst_bus_);

    if (!msg) break;

    DLOG(INFO) << "Get message, type=" << GST_MESSAGE_TYPE_NAME(msg);

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_EOS:
        DLOG(INFO) << "End of stream encountered";
        break;
      case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;

        gst_message_parse_error(msg, &error, &debug);
        g_free(debug);

        DLOG(INFO) << "GST error: %s\n", error->message;
        g_error_free(error);
        break;
      }
      case GST_MESSAGE_WARNING: {
        gchar *debug;
        GError *error;
        gst_message_parse_warning(msg, &error, &debug);
        g_free(debug);

        g_printerr("GST warning: %s\n", error->message);
        g_error_free(error);
        break;
      }
      case GST_MESSAGE_STATE_CHANGED: {
        GstState old_state, new_state;

        gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
        DLOG(INFO) << "Element " << GST_OBJECT_NAME(msg->src)
                   << " changed state from "
                   << gst_element_state_get_name(old_state) << " to "
                   << gst_element_state_get_name(new_state);
        break;
      }
      case GST_MESSAGE_STREAM_STATUS:
        GstStreamStatusType status;
        gst_message_parse_stream_status(msg, &status, NULL);
        switch (status) {
          case GST_STREAM_STATUS_TYPE_CREATE:
            DLOG(INFO) << "STREAM CREATED";
            break;
          case GST_STREAM_STATUS_TYPE_ENTER:
            DLOG(INFO) << "STREAM ENTERED";
            break;
          default:
            DLOG(INFO) << "OTHER STREAM STATUS " << status;
        }
        break;
      default:
        break;
    }

    gst_message_unref(msg);
  }

  return;
}
