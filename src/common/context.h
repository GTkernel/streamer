//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#ifndef TX1DNN_CONTEXT_H
#define TX1DNN_CONTEXT_H

#include <unordered_map>
#include "common.h"
using std::string;

const string H264_ENCODER_GST_ELEMENT = "h264_encoder_gst_element";
const string H264_DECODER_GST_ELEMENT = "h264_decoder_gst_element";
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
  Context() : config_dir_("./config") {}
  void Init() { GetEncoderDecoderInformation(); }

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

  void SetInt(const string &key, int value) { int_values_[key] = value; }

  void SetDouble(const string &key, double value) {
    double_values_[key] = value;
  }
  void SetString(const string &key, const string &value) {
    string_values_[key] = value;
  }

  void SetConfigDir(const string &config_dir) {
    config_dir_ = config_dir;
    // Reload the config
    Init();
  }

  string GetConfigDir() { return config_dir_; }

  string GetConfigFile(const string &filename) {
    return config_dir_ + "/" + filename;
  }

 private:
  /**
   * @brief Helper to initialze the context
   */
  void GetEncoderDecoderInformation() {
    auto root_value = ParseTomlFromFile(GetConfigFile("config.toml"));
    auto encoder_value = root_value.find("encoder");
    auto decoder_value = root_value.find("decoder");

    string_values_.insert(
        {H264_ENCODER_GST_ELEMENT,
         encoder_value->get<string>(H264_ENCODER_GST_ELEMENT)});
    string_values_.insert(
        {H264_DECODER_GST_ELEMENT,
         decoder_value->get<string>(H264_DECODER_GST_ELEMENT)});
  }

 private:
  string config_dir_;
  std::unordered_map<string, int> int_values_;
  std::unordered_map<string, string> string_values_;
  std::unordered_map<string, double> double_values_;
};

#endif