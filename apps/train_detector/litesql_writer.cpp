
#include "litesql_writer.h"

#include <sstream>
#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "camera/camera.h"
#include "framesdb.hpp"
#include "processor/compressor.h"
#include "processor/jpeg_writer.h"
#include "utils/file_utils.h"
#include "utils/time_utils.h"

constexpr auto SOURCE_NAME = "input";

LiteSqlWriter::LiteSqlWriter(const std::string& output_dir)
    : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {}),
      output_dir_(output_dir) {
  while (output_dir_.back() == '/') {
    output_dir_.pop_back();
  }
  // Create the output directory if it doesn't exist
  if (!CreateDirs(output_dir_)) {
    LOG(INFO) << "Using existing directory: \"" << output_dir_ << "\"";
  }
}

std::shared_ptr<LiteSqlWriter> LiteSqlWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<LiteSqlWriter>(params.at("output_dir"));
}

void LiteSqlWriter::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

bool LiteSqlWriter::Init() { return true; }

bool LiteSqlWriter::OnStop() { return true; }

void LiteSqlWriter::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  auto capture_time_micros =
      frame->GetValue<boost::posix_time::ptime>(Camera::kCaptureTimeMicrosKey);
  std::string data_dir =
      GetAndCreateDateTimeDir(output_dir_, capture_time_micros);

  std::string jpeg_path;
  if (frame->Count(JpegWriter::kPathKey)) {
    jpeg_path = frame->GetValue<std::string>(JpegWriter::kPathKey);
  }

  std::string compression_type;
  if (frame->Count(Compressor::kTypeKey)) {
    compression_type = frame->GetValue<std::string>(Compressor::kTypeKey);
  } else {
    compression_type =
        Compressor::CompressionTypeToString(Compressor::CompressionType::NONE);
  }

  std::ostringstream db_path;
  db_path << output_dir_ + "/frames.db";
  std::string db_path_str = db_path.str();
  FramesDB db("sqlite3", "database=" + db_path_str);
  std::string db_type;
  try {
    db.create();
    db_type = "new";
  } catch (litesql::Except e) {
    LOG(ERROR) << e.what();
    db_type = "existing";
  }
  LOG(INFO) << "Using " << db_type << " database: \"" << db_path_str << "\"";

  try {
    db.begin();
    db.verbose = true;

    FrameEntry fe(db);
    fe.dir = data_dir;
    fe.captureTimeMicros = GetDateTimeString(capture_time_micros);
    fe.jpegPath = jpeg_path;
    fe.compressionType = compression_type;
    fe.exposure = frame->GetValue<float>("CameraSettings.Exposure");
    fe.sharpness = frame->GetValue<float>("CameraSettings.Sharpness");
    fe.brightness = frame->GetValue<float>("CameraSettings.Brightness");
    fe.saturation = frame->GetValue<float>("CameraSettings.Saturation");
    fe.hue = frame->GetValue<float>("CameraSettings.Hue");
    fe.gain = frame->GetValue<float>("CameraSettings.Gain");
    fe.gamma = frame->GetValue<float>("CameraSettings.Gamma");
    fe.wbred = frame->GetValue<float>("CameraSettings.WBRed");
    fe.wbblue = frame->GetValue<float>("CameraSettings.WBBlue");
    fe.update();

    db.commit();
  } catch (litesql::Except e) {
    LOG(FATAL) << "\"" << db_path_str << "\""
               << " does not appear to be a valid sqlite3 database!"
               << std::endl
               << e.what();
  }
}
