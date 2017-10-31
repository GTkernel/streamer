
#include "processor/display.h"
#include "utils/image_utils.h"

#include <cstdio>

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Display::Display(std::string key, unsigned int angle, float zoom,
                 std::string window_name)
    : Processor(PROCESSOR_TYPE_DISPLAY, {SOURCE_NAME}, {SINK_NAME}),
      key_(key),
      angle_(angle),
      zoom_(zoom),
      window_name_(window_name) {
  CHECK(zoom >= 0 && zoom <= 1) << "Display zoom must be between 0 and 1";
  CHECK(angle == 0 || angle == 90 || angle == 180 || angle == 270)
      << "Display angle must be one of {0, 90, 180, 270}";
}

std::shared_ptr<Display> Display::Create(const FactoryParamsType& params) {
  std::string key = params.at("key");
  unsigned int angle = std::stoi(params.at("angle"));
  double zoom = std::stod(params.at("zoom"));
  std::string window_name = params.at("window_name");
  return std::make_shared<Display>(key, angle, zoom, window_name);
}

void Display::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Display::GetSink() { return Processor::GetSink(SINK_NAME); }

bool Display::Init() { return true; }

bool Display::OnStop() { return true; }

void Display::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  const cv::Mat& img = frame->GetValue<cv::Mat>(key_);
  cv::Mat m;
  if (zoom_ >= 0)
    cv::resize(img, m, cv::Size(), zoom_, zoom_);
  else
    cv::resize(img, m, cv::Size(), 1, 1);
  RotateImage(m, angle_);
  cv::imshow(window_name_, m);

  unsigned char q = cv::waitKey(10);

  PushFrame(SINK_NAME, std::move(frame));
}
