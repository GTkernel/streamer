
#include "processor/jpeg_writer.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/variant/get.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera.h"
#include "stream/frame.h"
#include "utils/time_utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

const char* JpegWriter::kPathKey = "JpegWriter.path";
const char* JpegWriter::kFieldKey = "JpegWriter.field";

JpegWriter::JpegWriter(const std::string& field, const std::string& output_dir,
                       bool organize_by_time, unsigned long frames_per_dir)
    : Processor(PROCESSOR_TYPE_JPEG_WRITER, {SOURCE_NAME}, {SINK_NAME}),
      field_(field),
      tracker_{output_dir, organize_by_time, frames_per_dir} {}

std::shared_ptr<JpegWriter> JpegWriter::Create(
    const FactoryParamsType& params) {
  std::string field = params.at("field");
  std::string output_dir = params.at("output_dir");
  bool organize_by_time = params.at("organize_by_time") == "1";
  unsigned long frames_per_dir = std::stoul(params.at("frames_per_dir"));
  return std::make_shared<JpegWriter>(field, output_dir, organize_by_time,
                                      frames_per_dir);
}

void JpegWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr JpegWriter::GetSink() { return Processor::GetSink(SINK_NAME); }

bool JpegWriter::Init() { return true; }

bool JpegWriter::OnStop() { return true; }

void JpegWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  cv::Mat img;
  try {
    img = frame->GetValue<cv::Mat>(field_);
  } catch (boost::bad_get& e) {
    LOG(FATAL) << "Unable to get field \"" << field_
               << "\" as a cv::Mat: " << e.what();
  }

  auto capture_time_micros =
      frame->GetValue<boost::posix_time::ptime>(Camera::kCaptureTimeMicrosKey);
  std::ostringstream filepath;
  filepath << tracker_.GetAndCreateOutputDir(capture_time_micros)
           << GetDateTimeString(capture_time_micros) << "_" << field_ << ".jpg";
  std::string filepath_s = filepath.str();
  try {
    // OpenCV expects float images to be normalized between 0 and 1,
    // and 8-bit pixel format to be 0-255.
    cv::Mat img_normalized;
    if (img.type() != CV_8UC3) {
      cv::normalize(img, img_normalized, 0, 1, cv::NORM_MINMAX);
    } else {
      img_normalized = img;
    }
    cv::imwrite(filepath_s, img_normalized);
  } catch (cv::Exception& e) {
    LOG(FATAL) << "Unable to write JPEG file \"" << filepath_s
               << "\": " << e.what();
  }

  frame->SetValue(kPathKey, filepath_s);
  frame->SetValue(kFieldKey, field_);
  PushFrame(SINK_NAME, std::move(frame));
};
