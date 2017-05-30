//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#ifndef STREAMER_CONTEXT_H
#define STREAMER_CONTEXT_H

#include <cppzmq/zmq.hpp>
#include <unordered_map>

#include "common.h"
#include "utils/utils.h"

const string H264_ENCODER_GST_ELEMENT = "h264_encoder_gst_element";
const string H264_DECODER_GST_ELEMENT = "h264_decoder_gst_element";
const string PUBLISHER_PORT = "publisher_port";
const string DEVICE_NUMBER = "device_number";
const string USEFP16 = "use_fp16";
const int DEVICE_NUMBER_CPU_ONLY = -1;
const int DEFAULT_PUBLISHER_PORT = 5536;
/**
 * @brief Global single context, used to store and access various global
 * information.
 */
class Context {
 public:
  /**
   * @brief Get singleton instance.
   */
  static Context &GetContext() {
    static Context context;
    return context;
  }

 public:
  Context()
      : config_dir_("./config"),
        zmq_context_{1},
        zmq_publisher_{zmq_context_, ZMQ_PUB} {}
  ~Context() {
    // Tear down the publisher socket
    zmq_publisher_.unbind(zmq_publisher_addr_);
  }
  void Init() {
    SetEncoderDecoderInformation();
    SetDefaultDeviceInformation();
    SetupZMQ();
    bool_values_.insert({USEFP16, false});
  }

  int GetInt(const string &key) {
    CHECK(int_values_.count(key) != 0) << "No integer value with key  " << key;
    return int_values_[key];
  }
  double GetDouble(const string &key) {
    CHECK(double_values_.count(key) != 0) << "No double value with key " << key;
    return double_values_[key];
  }
  string GetString(const string &key) {
    CHECK(string_values_.count(key) != 0) << "No string value with key " << key;
    return string_values_[key];
  }
  bool GetBool(const string &key) {
    CHECK(bool_values_.count(key) != 0) << "No bool value with key " << key;
    return bool_values_[key];
  }

  void SetInt(const string &key, int value) { int_values_[key] = value; }

  void SetDouble(const string &key, double value) {
    double_values_[key] = value;
  }
  void SetString(const string &key, const string &value) {
    string_values_[key] = value;
  }
  void SetBool(const string &key, bool value) { bool_values_[key] = value; }

  /**
   * @brief Reload the config dir, MUST call Init() after this.
   */
  void SetConfigDir(const string &config_dir) { config_dir_ = config_dir; }

  string GetConfigDir() { return config_dir_; }

  string GetConfigFile(const string &filename) {
    return config_dir_ + "/" + filename;
  }

  zmq::context_t &GetZMQContext() { return zmq_context_; }

  zmq::socket_t &GetZMQPublisher() { return zmq_publisher_; }

  int GetZMQPublisherPort() { return GetInt(PUBLISHER_PORT); }

 private:
  string ValidateEncoderElement(const string &encoder) {
    if (IsGstElementExists(encoder)) {
      return encoder;
    } else if (IsGstElementExists("vtenc_h264")) {
      return "vtenc_h264";
    } else if (IsGstElementExists("omxh264enc")) {
      return "omxh264enc";
    } else if (IsGstElementExists("x264enc")) {
      return "x264enc";
    } else if (IsGstElementExists("vaapih264enc")) {
      return "vaapih264enc";
    }

    LOG(WARNING) << "No known gst encoder element exists on the system";

    return "INVALID_ENCODER";
  }

  string ValidateDecoderElement(const string &decoder) {
    if (IsGstElementExists(decoder)) {
      return decoder;
    } else if (IsGstElementExists("avdec_h264")) {
      return "avdec_h264";
    } else if (IsGstElementExists("omxh264dec")) {
      return "omxh264dec";
    }

    LOG(WARNING) << "No known gst decoder element exists on the system";

    return "INVALID_DECODER";
  }
  /**
   * @brief Helper to initialze the context
   */
  void SetEncoderDecoderInformation() {
    string config_file = GetConfigFile("config.toml");

    auto root_value = ParseTomlFromFile(config_file);
    auto encoder_value = root_value.find("encoder");
    auto decoder_value = root_value.find("decoder");

    string encoder_element =
        encoder_value->get<string>(H264_ENCODER_GST_ELEMENT);
    string decoder_element =
        decoder_value->get<string>(H264_DECODER_GST_ELEMENT);

    string validated_encoder_element = ValidateEncoderElement(encoder_element);
    string validated_decoder_element = ValidateDecoderElement(decoder_element);

    if (validated_encoder_element != encoder_element) {
      LOG(WARNING) << "Using encoder " << validated_encoder_element
                   << " instead of " << encoder_element
                   << " from configuration";
      encoder_element = validated_encoder_element;
    }

    if (validated_decoder_element != decoder_element) {
      LOG(WARNING) << "using decoder " << validated_decoder_element
                   << " instead of " << decoder_element
                   << " from configuration";
      decoder_element = validated_decoder_element;
    }

    string_values_.insert({H264_ENCODER_GST_ELEMENT, encoder_element});
    string_values_.insert({H264_DECODER_GST_ELEMENT, decoder_element});
  }

  void SetDefaultDeviceInformation() {
    // Default, CPU only mode
    SetInt(DEVICE_NUMBER, DEVICE_NUMBER_CPU_ONLY);
  }

  // Set the zmq context that will be used globally.
  void SetupZMQ() {
    string config_file = GetConfigFile("config.toml");
    auto root_value = ParseTomlFromFile(config_file);
    auto port_value = root_value.find(PUBLISHER_PORT);
    if (port_value == nullptr) {
      LOG(INFO) << "Using default publisher port " << DEFAULT_PUBLISHER_PORT;
      SetInt(PUBLISHER_PORT, DEFAULT_PUBLISHER_PORT);
    } else {
      LOG(INFO) << "Using publisher port: " << port_value->as<int>();
      SetInt(PUBLISHER_PORT, port_value->as<int>());
    }

    // Bind the publisher socket
    zmq_publisher_addr_ =
        "tcp://127.0.0.1:" + std::to_string(GetInt(PUBLISHER_PORT));
    LOG(INFO) << zmq_publisher_addr_;
    zmq_publisher_.bind(zmq_publisher_addr_);
  }

 private:
  string config_dir_;

  zmq::context_t zmq_context_;
  zmq::socket_t zmq_publisher_;

  std::unordered_map<string, int> int_values_;
  std::unordered_map<string, string> string_values_;
  std::unordered_map<string, double> double_values_;
  std::unordered_map<string, bool> bool_values_;

  string zmq_publisher_addr_;
};

#endif
