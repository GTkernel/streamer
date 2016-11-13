//
// Created by Ran Xian (xranthoar@gmail.com) on 11/2/16.
//

#include "gige_file_writer.h"

#include <boost/filesystem.hpp>

GigeFileWriter::GigeFileWriter(const string &directory, size_t frames_per_file)
    : Processor({"input"}, {}),
      directory_(directory),
      frames_written_(0),
      frames_per_file_(frames_per_file) {}

bool GigeFileWriter::Init() {
  // Make a directory for this run
  if (!boost::filesystem::exists(directory_)) {
    boost::filesystem::create_directory(directory_);
  } else {
    LOG(WARNING) << "Directory: " << directory_
                 << " already exists, may re-write existed files";
  }

  frames_written_ = 0;
  current_filename_ = "";

  return true;
}

bool GigeFileWriter::OnStop() {
  if (current_file_.is_open()) {
    current_file_.close();
  }
  return true;
}

void GigeFileWriter::Process() {
  // Create file
  if (frames_written_ % frames_per_file_ == 0) {
    std::ostringstream ss;
    ss << frames_written_ / frames_per_file_ << ".dat";

    string filename = directory_ + "/" + ss.str();

    if (current_file_.is_open()) current_file_.close();

    current_file_.open(filename, std::ios::binary | std::ios::out);

    if (!current_file_.is_open()) {
      LOG(FATAL) << "Can't open file: " << filename << " for write";
    }

    current_filename_ = filename;
  }

  auto frame = GetFrame<BytesFrame>("input");

  current_file_.write((char *)frame->GetDataBuffer().GetBuffer(),
                      frame->GetDataBuffer().GetSize());

  frames_written_ += 1;
}

ProcessorType GigeFileWriter::GetType() { return PROCESSOR_TYPE_CUSTOM; }
