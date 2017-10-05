
#include "processor/jpeg_writer.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/variant/get.hpp>
#include <opencv2/opencv.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

JpegWriter::JpegWriter(const std::string& key, const std::string& output_dir,
                       bool organize_by_time, unsigned long frames_per_dir)
    : Processor(PROCESSOR_TYPE_JPEG_WRITER, {SOURCE_NAME}, {}),
      key_(key),
      tracker_{output_dir, organize_by_time, frames_per_dir} {}

std::shared_ptr<JpegWriter> JpegWriter::Create(
    const FactoryParamsType& params) {
  std::string key = params.at("key");
  std::string output_dir = params.at("output_dir");
  bool organize_by_time = params.at("organize_by_time") == "1";
  unsigned long frames_per_dir = std::stoul(params.at("frames_per_dir"));
  return std::make_shared<JpegWriter>(key, output_dir, organize_by_time,
                                      frames_per_dir);
}

void JpegWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool JpegWriter::Init() { return true; }

bool JpegWriter::OnStop() { return true; }

void JpegWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  auto capture_time_micros =
      frame->GetValue<boost::posix_time::ptime>("capture_time_micros");
  auto id = frame->GetValue<unsigned long>("frame_id");
  std::ostringstream filepath;
  filepath << tracker_.GetAndCreateOutputDir(capture_time_micros) << id << "_"
           << boost::posix_time::to_iso_extended_string(capture_time_micros)
           << ".jpg";
  std::string filepath_s = filepath.str();

  cv::Mat img;
  try {
    img = frame->GetValue<cv::Mat>(key_);
  } catch (boost::bad_get& e) {
    LOG(FATAL) << "Unable to get key \"" << key_
               << "\" as a cv::Mat: " << e.what();
  } catch (std::out_of_range& e) {
    LOG(FATAL) << "Key \"" << key_ << "\" not in frame.";
  }
  try {
    cv::imwrite(filepath_s, img);
  } catch (cv::Exception& e) {
    LOG(FATAL) << "Unable to write JPEG file \"" << filepath_s
               << "\": " << e.what();
  }
};
