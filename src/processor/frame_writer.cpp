
#include "processor/frame_writer.h"

#include <sstream>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

FrameWriter::FrameWriter(const std::unordered_set<std::string> fields,
                         const std::string& output_dir, const FileFormat format)
    : Processor(PROCESSOR_TYPE_FRAME_WRITER, {SOURCE_NAME}, {}),
      fields_(fields),
      output_dir_(output_dir),
      format_(format) {}

std::shared_ptr<FrameWriter> FrameWriter::Create(
    const FactoryParamsType& params) {
  std::unordered_set<std::string> fields;
  // TODO: Parse field names. Currently, FactoryParamsType does not support keys
  //       that are sets.

  std::string output_dir = params.at("output_dir");

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

  return std::make_shared<FrameWriter>(fields, output_dir, format);
}

void FrameWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool FrameWriter::Init() { return true; }

bool FrameWriter::OnStop() { return true; }

void FrameWriter::Process() {
  if (!boost::filesystem::exists(output_dir_)) {
    LOG(FATAL) << "Directory \"" << output_dir_ << "\" does not exist.";
  }

  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  std::stringstream filepath;
  auto id = frame->GetValue<unsigned long>("frame_id");
  filepath << output_dir_ << "/" << id << GetExtension();
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
}

std::string FrameWriter::GetExtension() {
  switch (format_) {
    case BINARY:
      return ".bin";
    case JSON:
      return ".json";
    case TEXT:
      return ".txt";
  }

  LOG(FATAL) << "Unhandled FileFormat: " << format_;
}