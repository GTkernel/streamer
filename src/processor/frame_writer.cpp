
#include "processor/frame_writer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

FrameWriter::FrameWriter(const std::unordered_set<std::string> fields,
                         const std::string& output_dir,
                         unsigned int frames_per_dir, const FileFormat format)
    : Processor(PROCESSOR_TYPE_FRAME_WRITER, {SOURCE_NAME}, {}),
      fields_(fields),
      output_dir_(output_dir),
      frames_per_dir_(frames_per_dir),
      frames_in_current_dir_(0),
      current_dir_(0),
      format_(format) {
  if (!boost::filesystem::exists(output_dir_)) {
    LOG(FATAL) << "Directory \"" << output_dir_ << "\" does not exist.";
  }

  SetSubdir(current_dir_);
}

std::shared_ptr<FrameWriter> FrameWriter::Create(
    const FactoryParamsType& params) {
  // TODO: Parse field names. Currently, FactoryParamsType does not support keys
  //       that are sets.
  std::unordered_set<std::string> fields;

  std::string output_dir = params.at("output_dir");

  int frames_per_dir_int = std::stoi(params.at("frames_per_dir"));
  if (frames_per_dir_int < 0) {
    throw std::invalid_argument(
        "\"frame_per_dir\" must be greater than 0, but is: " +
        std::to_string(frames_per_dir_int));
  }

  std::string format_s = params.at("format");
  FileFormat format;
  if (format_s == "binary") {
    format = BINARY;
  } else if (format_s == "json") {
    format = JSON;
  } else if (format_s == "text") {
    format = TEXT;
  } else {
    LOG(FATAL) << "Unknown file format: " << format_s;
  }

  return std::make_shared<FrameWriter>(
      fields, output_dir, (unsigned int)frames_per_dir_int, format);
}

void FrameWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool FrameWriter::Init() { return true; }

bool FrameWriter::OnStop() { return true; }

void FrameWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  auto id = frame->GetValue<unsigned long>("frame_id");

  std::stringstream filepath;
  filepath << output_subdir_ << "/" << id << "." << GetExtension();
  std::string filepath_s = filepath.str();
  std::ofstream file(filepath_s, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    LOG(FATAL) << "Unable to open file \"" << filepath_s << "\".";
  }

  auto frame_to_write = std::make_unique<Frame>(frame, fields_);
  try {
    switch (format_) {
      case BINARY: {
        boost::archive::binary_oarchive ar(file);
        ar << frame_to_write;
        break;
      }
      case JSON: {
        file << frame_to_write->ToJson().dump(4);
        break;
      }
      case TEXT: {
        boost::archive::text_oarchive ar(file);
        ar << frame_to_write;
        break;
      }
    }
  } catch (const boost::archive::archive_exception& e) {
    LOG(FATAL) << "Boost serialization error: " << e.what();
  }

  file.close();
  if (!file) {
    LOG(FATAL) << "Unknown error while writing binary file \"" << filepath_s
               << "\".";
  }

  // If we have filled up the current subdir, then move on to the next one.
  ++frames_in_current_dir_;
  if (frames_in_current_dir_ == frames_per_dir_) {
    frames_in_current_dir_ = 0;
    ++current_dir_;
    SetSubdir(current_dir_);
  }
}

std::string FrameWriter::GetExtension() {
  switch (format_) {
    case BINARY:
      return "bin";
    case JSON:
      return "json";
    case TEXT:
      return "txt";
  }

  LOG(FATAL) << "Unhandled FileFormat: " << format_;
}

void FrameWriter::SetSubdir(unsigned int subdir) {
  current_dir_ = subdir;
  output_subdir_ = output_dir_ + std::to_string(current_dir_);

  std::string current_dir_str = output_dir_ + std::to_string(current_dir_);
  boost::filesystem::path current_dir_path(current_dir_str);
  boost::filesystem::create_directory(current_dir_path);
}
