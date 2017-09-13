
#include "processor/jpeg_writer.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/variant/get.hpp>
#include <opencv2/opencv.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

JpegWriter::JpegWriter(std::string key, std::string filepath,
                       unsigned long num_frames_per_dir)
    : Processor(PROCESSOR_TYPE_JPEG_WRITER, {SOURCE_NAME}, {}),
      key_(key),
      output_dir_(filepath),
      num_frames_per_dir_(num_frames_per_dir) {}

std::shared_ptr<JpegWriter> JpegWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<JpegWriter>(
      params.at("key"), params.at("output_dir"),
      std::stoul(params.at("num_frames_per_dir")));
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
  auto id = frame->GetValue<unsigned long>("frame_id");

  std::ostringstream dirpath;
  auto dir_num = id / num_frames_per_dir_;
  dirpath << output_dir_ << "/" << dir_num;
  auto dirpath_str = dirpath.str();
  boost::filesystem::path dir(dirpath_str);
  boost::filesystem::create_directory(dir);

  std::stringstream filepath;
  filepath << dirpath_str << "/" << id << ".jpg";
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
