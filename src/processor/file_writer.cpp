//
// Created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//

#include "file_writer.h"
#include "utils/file_utils.h"

FileWriter::FileWriter(const string& filename)
    : Processor({"input"}, {}), filename_(filename) {}

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
  auto frame = GetFrame<Frame>("input");
  switch (frame->GetType()) {
    case FRAME_TYPE_BYTES: {
      auto bytes_frame = std::dynamic_pointer_cast<BytesFrame>(frame);
      auto buffer = bytes_frame->GetDataBuffer();
      file_.write((char*)buffer.GetBuffer(), buffer.GetSize());
      break;
    }
    case FRAME_TYPE_IMAGE: {
      auto image_frame = std::dynamic_pointer_cast<ImageFrame>(frame);
      auto image = image_frame->GetImage();
      file_.write((char*)image.data, image.total() * image.elemSize());
      break;
    }
    case FRAME_TYPE_MD: {
      auto md_frame = std::dynamic_pointer_cast<MetadataFrame>(frame);
      std::string s = md_frame->ToJson().dump();
      file_.write(s.c_str(), sizeof(char) * s.length());
      break;
    }
    default: { STREAMER_NOT_IMPLEMENTED; }
  }
}

ProcessorType FileWriter::GetType() const { return PROCESSOR_TYPE_FILE_WRITER; }
