//
// Created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//

#include "file_writer.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "utils/file_utils.h"

FileWriter::FileWriter(const string& filename, const file_format format)
    : Processor(PROCESSOR_TYPE_FILE_WRITER, {"input"}, {}),
      filename_(filename),
      format_(format) {}

// TODO: Fix Create to accept second argument for file format
std::shared_ptr<FileWriter> FileWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<FileWriter>(params.at("filename"));
}

bool FileWriter::Init() {
  // Create the directory of the file if not exist
  CreateDirs(GetDir(filename_));

  file_.open(filename_, std::ios::binary);

  return file_.is_open();
}

bool FileWriter::OnStop() {
  if (file_.is_open()) {
    file_.close();
  }

  return true;
}

void FileWriter::Process() {
  auto frame = GetFrame("input");

  try {
    switch (format_) {
      case TEXT: {
        boost::archive::text_oarchive ar(file_);
        ar << frame;
        break;
      }
      case BINARY: {
        boost::archive::binary_oarchive ar(file_);
        ar << frame;
        break;
      }
      default: { return; }
    }
  } catch (const boost::archive::archive_exception& e) {
    LOG(INFO) << "Boost serialization error: " << e.what();
  }
}
