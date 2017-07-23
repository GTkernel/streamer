
#include "processor/jpeg_writer.h"

#include <sstream>

#include <opencv2/opencv.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

JpegWriter::JpegWriter(std::string key, std::string filepath)
    : Processor(PROCESSOR_TYPE_JPEG_WRITER, {SOURCE_NAME}, {}),
      key_(key),
      output_dir_(filepath) {}

std::shared_ptr<JpegWriter> JpegWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<JpegWriter>(params.at("key"),
                                      params.at("output_dir"));
}

void JpegWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool JpegWriter::Init() { return true; }

bool JpegWriter::OnStop() { return true; }

void JpegWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  if (!frame->Count(key_)) LOG(FATAL) << "Key \"" << key_ << "\" not in frame.";

  std::stringstream filepath;
  auto id = frame->GetValue<unsigned long>("frame_id");
  filepath << output_dir_ << "/" << id << ".jpg";

  const auto& img = frame->GetValue<cv::Mat>(key_);
  cv::imwrite(filepath.str(), img);
};
