
#include "processor/jpeg_writer.h"

#include <sstream>

#include <boost/filesystem.hpp>
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
  if (!boost::filesystem::exists(output_dir_)) {
    LOG(FATAL) << "Directory \"" << output_dir_ << "\" does not exist.";
  }

  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  std::stringstream filepath;
  auto id = frame->GetValue<unsigned long>("frame_id");
  filepath << output_dir_ << "/" << id << ".jpg";
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
