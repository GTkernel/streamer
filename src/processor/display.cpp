
#include "processor/display.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "stream/frame.h"
#include "utils/image_utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Display::Display(const std::string& key, unsigned int angle, float size_ratio,
                 const std::string& window_name)
    : Processor(PROCESSOR_TYPE_DISPLAY, {SOURCE_NAME}, {SINK_NAME}),
      key_(key),
      angle_(angle),
      size_ratio_(size_ratio),
      window_name_(window_name) {
  std::unordered_set<unsigned int> possible_angles = {0, 90, 180, 270};
  if (possible_angles.find(angle_) == possible_angles.end()) {
    std::ostringstream msg;
    msg << "\"angle\" must be one of { ";
    for (auto possible_angle : possible_angles) {
      msg << possible_angle << " ";
    }
    msg << "}, but is: " << angle_;
    throw std::invalid_argument(msg.str());
  }
  if (size_ratio_ < 0 || size_ratio_ > 1) {
    std::ostringstream msg;
    msg << "\"size_ratio\" must be in the range [0, 1], but is: "
        << size_ratio_;
    throw std::invalid_argument(msg.str());
  }
}

std::shared_ptr<Display> Display::Create(const FactoryParamsType& params) {
  std::string key = params.at("key");
  int angle = std::stoi(params.at("angle"));
  if (angle < 0) {
    std::ostringstream msg;
    msg << "\"angle\" cannot be negative, but is: " << angle;
    throw std::invalid_argument(msg.str());
  }
  double size_ratio = std::stod(params.at("size_ratio"));
  std::string window_name = params.at("window_name");
  return std::make_shared<Display>(key, (unsigned int)angle, size_ratio,
                                   window_name);
}

void Display::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Display::GetSink() { return Processor::GetSink(SINK_NAME); }

bool Display::Init() { return true; }

bool Display::OnStop() { return true; }

void Display::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  const cv::Mat& img = frame->GetValue<cv::Mat>(key_);

  cv::Mat display_img;
  if (size_ratio_ >= 0) {
    cv::resize(img, display_img, cv::Size(), size_ratio_, size_ratio_);
  } else {
    display_img = img;
  }
  RotateImage(display_img, angle_);

  cv::imshow(window_name_, display_img);
  cv::waitKey(10);

  PushFrame(SINK_NAME, std::move(frame));
}
