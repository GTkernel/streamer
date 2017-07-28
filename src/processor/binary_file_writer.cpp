
#include "processor/binary_file_writer.h"

#include <fstream>
#include <iostream>
#include <sstream>

constexpr auto SOURCE_NAME = "input";

BinaryFileWriter::BinaryFileWriter(std::string key, std::string output_dir)
    : Processor(PROCESSOR_TYPE_BINARY_FILE_WRITER, {SOURCE_NAME}, {}),
      key_(key),
      output_dir_(output_dir) {}

std::shared_ptr<BinaryFileWriter> BinaryFileWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<BinaryFileWriter>(params.at("key"),
                                            params.at("output_dir"));
}

void BinaryFileWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool BinaryFileWriter::Init() { return true; }

bool BinaryFileWriter::OnStop() { return true; }

void BinaryFileWriter::Process() {
  if (!boost::filesystem::exists(output_dir_)) {
    LOG(FATAL) << "Directory \"" << output_dir_ << "\" does not exist.";
  }

  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  std::stringstream filepath;
  auto id = frame->GetValue<unsigned long>("frame_id");
  filepath << output_dir_ << "/" << id << ".bin";
  std::string filepath_s = filepath.str();
  std::ofstream file(filepath_s, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    LOG(FATAL) << "Unable to open file \"" << filepath_s << "\".";
  }

  std::vector<char> bytes;
  try {
    bytes = frame->GetValue<std::vector<char>>(key_);
  } catch (boost::bad_get& e) {
    LOG(FATAL) << "Unable to get key \"" << key_
               << "\" as an std::vector<char>: " << e.what();
  } catch (std::out_of_range& e) {
    LOG(FATAL) << "Key \"" << key_ << "\" not in frame.";
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
