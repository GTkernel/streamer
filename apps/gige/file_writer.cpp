//
// Created by Ran Xian (xranthoar@gmail.com) on 11/2/16.
//

#include "file_writer.h"
#include "boost/filesystem.hpp"

FileWriter::FileWriter(const string &filename_base, size_t frames_per_file)
    : Processor({"input"}, {}),
      filename_base_(filename_base),
      frames_written_(0),
      frames_per_file_(frames_per_file) {}

bool FileWriter::Init() {
  // Make a directory for this run
  directory_name_ = GetCurrentTimeString("streamer-%Y%m%d-%H%M%S");
  if (!boost::filesystem::exists(directory_name_))
    boost::filesystem::create_directory(directory_name_);

  frames_written_ = 0;
  current_filename_ = "";

  return true;
}

bool FileWriter::OnStop() {
  if (current_file_.is_open()) {
    current_file_.close();
  }
  return true;
}

void FileWriter::Process() {
  // Create file
  if (frames_written_ % frames_per_file_ == 0) {
    std::ostringstream ss;
    if (filename_base_ != "") {
      ss << filename_base_ << "-";
    }
    ss << frames_written_ / frames_per_file_ << ".dat";

    string filename = directory_name_ + "/" + ss.str();

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

ProcessorType FileWriter::GetType() { return PROCESSOR_TYPE_CUSTOM; }
