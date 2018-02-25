
#include "processor/binary_file_writer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "camera/camera.h"
#include "utils/time_utils.h"

constexpr auto SOURCE_NAME = "input";

BinaryFileWriter::BinaryFileWriter(const std::string& field,
                                   const std::string& output_dir,
                                   bool organize_by_time,
                                   unsigned long frames_per_dir)
    : Processor(PROCESSOR_TYPE_BINARY_FILE_WRITER, {SOURCE_NAME}, {}),
      field_(field),
      tracker_{output_dir, organize_by_time, frames_per_dir} {}

std::shared_ptr<BinaryFileWriter> BinaryFileWriter::Create(
    const FactoryParamsType& params) {
  std::string field = params.at("field");
  std::string output_dir = params.at("output_dir");
  bool organize_by_time = params.at("organize_by_time") == "1";
  unsigned long frames_per_dir = std::stoul(params.at("frames_per_dir"));
  return std::make_shared<BinaryFileWriter>(field, output_dir, organize_by_time,
                                            frames_per_dir);
}

void BinaryFileWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool BinaryFileWriter::Init() { return true; }

bool BinaryFileWriter::OnStop() { return true; }

void BinaryFileWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  auto capture_time_micros =
      frame->GetValue<boost::posix_time::ptime>(Camera::kCaptureTimeMicrosKey);
  std::ostringstream filepath;
  filepath << tracker_.GetAndCreateOutputDir(capture_time_micros)
           << GetDateTimeString(capture_time_micros) << "_" << field_ << ".bin";
  std::string filepath_s = filepath.str();
  std::ofstream file(filepath_s, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    LOG(FATAL) << "Unable to open file \"" << filepath_s << "\".";
  }

  std::vector<char> bytes;
  try {
    bytes = frame->GetValue<std::vector<char>>(field_);
  } catch (boost::bad_get& e) {
    LOG(FATAL) << "Unable to get field \"" << field_
               << "\" as an std::vector<char>: " << e.what();
  }
  try {
    file.write((char*)bytes.data(), bytes.size());
    file.close();
    if (!file) {
      LOG(FATAL) << "Unknown error while writing binary file \"" << filepath_s
                 << "\".";
    }
  } catch (std::ios_base::failure& e) {
    LOG(FATAL) << "Error while writing binary \"" << filepath_s
               << "\": " << e.what();
  }
}
