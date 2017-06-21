//
// Created by Ran Xian (xranthoar@gmail.com) on 10/13/16.
//

#include "gst_video_encoder.h"

#include <stdlib.h>

#include "common/context.h"
#include "utils/string_utils.h"

const static char* ENCODER_SRC_NAME = "encoder_src";

GstVideoEncoder::GstVideoEncoder(int width, int height,
                                 const string& output_filename)
    : Processor(PROCESSOR_TYPE_ENCODER, {"input"}, {"output"}),
      width_(width),
      height_(height),
      frame_size_bytes_(width * height * 3),
      port_(-1),  // Not in streaming mode
      output_filename_(output_filename),
      need_data_(false),
      timestamp_(0) {
  CHECK(width > 0 && height > 0) << "Width or height is invalid";
  // Encoder
  encoder_element_ = Context::GetContext().GetString(H264_ENCODER_GST_ELEMENT);
}

GstVideoEncoder::GstVideoEncoder(int width, int height, int port, bool tcp)
    : Processor(PROCESSOR_TYPE_ENCODER, {"input"}, {"output"}),
      width_(width),
      height_(height),
      frame_size_bytes_(width * height * 3),
      port_(port),
      tcp_(tcp),
      need_data_(false),
      timestamp_(0) {
  CHECK(width > 0 && height > 0) << "Width or height is invalid";
  // Encoder
  encoder_element_ = Context::GetContext().GetString(H264_ENCODER_GST_ELEMENT);
}

std::shared_ptr<GstVideoEncoder> GstVideoEncoder::Create(
    const FactoryParamsType& params) {
  int port = -1;
  string filename;

  if (params.count("port") != 0) {
    port = atoi(params.at("port").c_str());
  } else if (params.count("filename") != 0) {
    filename = params.at("filename");
  } else {
    LOG(FATAL) << "At least port or filename is needed for encoder";
  }

  int width = atoi(params.at("width").c_str());
  int height = atoi(params.at("height").c_str());
  CHECK(width >= 0 && height >= 0) << "Width (" << width << ") and height ("
                                   << height << ") must not be negative.";

  if (port > 0) {
    return std::make_shared<GstVideoEncoder>(width, height, port);
  } else {
    return std::make_shared<GstVideoEncoder>(width, height, filename);
  }
}

void GstVideoEncoder::NeedDataCB(GstAppSrc*, guint, gpointer user_data) {
  if (user_data == nullptr) return;

  GstVideoEncoder* encoder = (GstVideoEncoder*)user_data;
  if (encoder->IsStarted()) encoder->need_data_ = true;
}

void GstVideoEncoder::EnoughDataCB(GstAppSrc*, gpointer user_data) {
  DLOG(INFO) << "Received enough data signal";
  if (user_data == nullptr) return;

  GstVideoEncoder* encoder = (GstVideoEncoder*)user_data;
  encoder->need_data_ = false;
}

/**
 * @brief Build the encoder pipeline. We will create a pipeline to store to a
 * file if ouput_filename_ is not empty, or a pipeline to stream the video
 * through a udp port if port_ is not empty.
 */
string GstVideoEncoder::BuildPipelineString() {
  std::ostringstream ss;

  ss << "appsrc name=" << ENCODER_SRC_NAME << " ! "
     << "videoconvert ! " << encoder_element_ << " ! ";

  if (output_filename_ != "" && port_ != -1) {
    ss << "tee name=t ! ";
  }

  if (output_filename_ != "") {
    ss << "qtmux ! filesink location=" << output_filename_;
    if (port_ != -1) ss << "t. ! ";
  }

  if (port_ != -1) {
    if (tcp_) {
      ss << "mpegtsmux ! tcpserversink port=" << port_;
    } else {
      ss << "rtph264pay config-interval=1 ! "
         << "udpsink host=127.0.0.1 port=" << port_ << " auto-multicast=true";
    }
  }

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
  // Create a directory for file if the directory does not exist yet
  if (output_filename_ != "") {
    CreateDirs(GetDir(output_filename_));
  }

  if (width_ == 0 || height_ == 0) {
    LOG(ERROR) << "width or height of output video is not valid (" << width_
               << "x" << height_ << ")";
    return false;
  }

  g_main_loop_ = g_main_loop_new(NULL, FALSE);

  // Build Gst pipeline
  GError* err = nullptr;
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

  GstElement* appsrc_element =
      gst_bin_get_by_name(GST_BIN(gst_pipeline_), ENCODER_SRC_NAME);
  GstAppSrc* appsrc = GST_APP_SRC(appsrc_element);

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

  gst_app_src_set_caps(gst_appsrc_, gst_caps_);
  g_object_set(G_OBJECT(gst_appsrc_), "stream-type", 0, "format",
               GST_FORMAT_TIME, NULL);

  GstAppSrcCallbacks callbacks;
  callbacks.enough_data = GstVideoEncoder::EnoughDataCB;
  callbacks.need_data = GstVideoEncoder::NeedDataCB;
  callbacks.seek_data = NULL;
  gst_app_src_set_callbacks(gst_appsrc_, &callbacks, (void*)this, NULL);

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
  auto input_frame = GetFrame("input");

  // Resize the image
  cv::Mat image;
  cv::Mat original_image = input_frame->GetOriginalImage();

  cv::resize(original_image, image, cv::Size(width_, height_));

  Frame* image_frame = new Frame();

  image_frame->SetOriginalImage(original_image);
  image_frame->SetImage(image);

  // Forward the input image to output
  PushFrame("output", image_frame);

  // Lock the state of the encoder
  std::lock_guard<std::mutex> guard(encoder_lock_);

  if (!need_data_) return;

  // Give PTS to the buffer
  GstMapInfo info;
  GstBuffer* buffer =
      gst_buffer_new_allocate(nullptr, frame_size_bytes_, nullptr);
  gst_buffer_map(buffer, &info, GST_MAP_WRITE);
  // Copy the image to gst buffer, should have better way such as using
  // gst_buffer_new_wrapper_full(). TODO
  memcpy(info.data, image.data, frame_size_bytes_);
  gst_buffer_unmap(buffer, &info);

  GST_BUFFER_PTS(buffer) = timestamp_;

  // TODO: FPS is fixed right now
  GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, 30);
  timestamp_ += GST_BUFFER_DURATION(buffer);

  GstFlowReturn ret;
  g_signal_emit_by_name(gst_appsrc_, "push-buffer", buffer, &ret);

  gst_buffer_unref(buffer);

  if (ret != 0) {
    LOG(INFO) << "Encoder -- appsrc can't push buffer, ret code (" << ret
              << ")";
  }

  // Poll messages from the bus
  while (true) {
    GstMessage* msg = gst_bus_pop(gst_bus_);

    if (!msg) break;

    DLOG(INFO) << "Get message, type=" << GST_MESSAGE_TYPE_NAME(msg);

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_EOS:
        DLOG(INFO) << "End of stream encountered";
        break;
      case GST_MESSAGE_ERROR: {
        gchar* debug;
        GError* error;

        gst_message_parse_error(msg, &error, &debug);
        g_free(debug);

        DLOG(WARNING) << "GST error: " << error->message;
        g_error_free(error);
        break;
      }
      case GST_MESSAGE_WARNING: {
        gchar* debug;
        GError* error;
        gst_message_parse_warning(msg, &error, &debug);
        g_free(debug);

        DLOG(WARNING) << "GST warning: " << error->message;
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

void GstVideoEncoder::SetEncoderElement(const string& encoder) {
  encoder_element_ = encoder;
}
