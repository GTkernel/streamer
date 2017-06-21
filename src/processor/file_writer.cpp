//
// Created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//

#include "file_writer.h"

#include "utils/file_utils.h"

FileWriter::FileWriter(const string& filename)
    : Processor(PROCESSOR_TYPE_FILE_WRITER, {"input"}, {}),
      filename_(filename) {}

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
  switch (frame->GetType()) {
    case FRAME_TYPE_BYTES: {
      auto buffer = frame->GetValue<DataBuffer>("DataBuffer");
      file_.write((char*)buffer.GetBuffer(), buffer.GetSize());
      break;
    }
    case FRAME_TYPE_IMAGE: {
      auto image = frame->GetValue<cv::Mat>("Image");
      file_.write((char*)image.data, image.total() * image.elemSize());
      break;
    }
    case FRAME_TYPE_MD: {
      std::string s = frame->ToJson().dump();
      file_.write(s.c_str(), sizeof(char) * s.length());
      break;
    }
    default: { STREAMER_NOT_IMPLEMENTED; }
  }
}
