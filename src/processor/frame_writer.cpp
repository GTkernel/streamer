
#include "processor/frame_writer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

#include "stream/frame.h"

constexpr auto SOURCE_NAME = "input";

FrameWriter::FrameWriter(const std::unordered_set<std::string> fields,
                         const std::string& output_dir, const FileFormat format,
                         bool save_fields_separately,
                         unsigned int frames_per_dir, bool organize_by_time)
    : Processor(PROCESSOR_TYPE_FRAME_WRITER, {SOURCE_NAME}, {}),
      fields_(fields),
      output_dir_(output_dir),
      format_(format),
      save_fields_separately_(save_fields_separately),
      organize_by_time_(organize_by_time),
      output_subdir_(""),
      frames_per_dir_(frames_per_dir),
      frames_in_current_dir_(0),
      current_dir_(0) {
  if (!organize_by_time) {
    SetSubdir(current_dir_);
  }
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

  return std::make_shared<FrameWriter>(fields, output_dir, format, false,
                                       (unsigned int)frames_per_dir_int);
}

void FrameWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool FrameWriter::Init() { return true; }

bool FrameWriter::OnStop() { return true; }

void FrameWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  auto frame_to_write = std::make_unique<Frame>(frame, fields_);

  // Accumulates the path at which this frame will be stored.
  std::ostringstream base_filepath;

  auto capture_time_micros =
      frame->GetValue<boost::posix_time::ptime>("capture_time_micros");
  if (organize_by_time_) {
    // Add subdirectories for date and time.
    std::string date_s =
        boost::gregorian::to_iso_extended_string(capture_time_micros.date());
    long hours = capture_time_micros.time_of_day().hours();
    base_filepath << output_dir_ << "/" << date_s << "/" << hours << "/";
    // Create the output directory, since it might not exist yet.
    boost::filesystem::create_directories(
        boost::filesystem::path{base_filepath.str()});
  } else {
    base_filepath << output_subdir_ << "/";
  }

  // The filepath always includes the frame id and the capture time.
  auto id = frame->GetValue<unsigned long>("frame_id");
  base_filepath << id << "_"
                << boost::posix_time::to_iso_extended_string(
                       capture_time_micros);

  if (save_fields_separately_) {
    // Create a separate file for each field.
    for (const auto& p : frame_to_write->GetFields()) {
      std::string key = p.first;
      Frame::field_types value = p.second;

      // The final filepath needs the key and an extension.
      std::ostringstream filepath;
      filepath << base_filepath.str() << "_" << key << GetExtension();

      std::string filepath_s = filepath.str();
      std::ofstream file(filepath_s, std::ios::binary | std::ios::out);
      if (!file.is_open()) {
        LOG(FATAL) << "Unable to open file \"" << filepath_s << "\".";
      }

      try {
        switch (format_) {
          case BINARY: {
            boost::archive::binary_oarchive ar(file);
            ar << value;
            break;
          }
          case JSON: {
            file << frame_to_write->GetFieldJson(key).dump(4);
            break;
          }
          case TEXT: {
            boost::archive::text_oarchive ar(file);
            ar << value;
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
  } else {
    // The final filepath just needs an extension.
    base_filepath << GetExtension();

    std::string filepath_s = base_filepath.str();
    std::ofstream file(filepath_s, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
      LOG(FATAL) << "Unable to open file \"" << filepath_s << "\".";
    }

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

  if (!organize_by_time_) {
    // If we have filled up the current subdir, then move on to the next one.
    ++frames_in_current_dir_;
    if (frames_in_current_dir_ > frames_per_dir_) {
      frames_in_current_dir_ = 0;
      ++current_dir_;
      SetSubdir(current_dir_);
    }
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

void FrameWriter::SetSubdir(unsigned int subdir) {
  current_dir_ = subdir;
  output_subdir_ = output_dir_ + std::to_string(current_dir_);

  std::ostringstream current_dir;
  current_dir << output_dir_ << "/" << current_dir_;
  output_subdir_ = current_dir.str();
  boost::filesystem::create_directory(boost::filesystem::path{output_subdir_});
}
