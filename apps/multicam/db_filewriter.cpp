//
// Not created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//
#include "db_filewriter.h"

#include <chrono>
#include <fstream>

#include "json/json.hpp"
#include "utils/file_utils.h"
#include "framesdatabase.hpp"
#include "sys/stat.h"

// TODO: this will change to take in the frame after the metadata is added to the frame
void DoWriteDB(std::string filename, time_t cur_time, std::unique_ptr<Frame>& frame) {
#ifdef USE_LITESQL
  // assumes database has been created
    FramesDatabase db("sqlite3", "database=frames.db");
    try {
      db.create();
    } catch (...) { }
    db.verbose = true;
    FrameEntry fe(db);
    fe.path = filename;
    fe.date = cur_time;
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
#endif // USE_LITESQL
}

DBFileWriter::DBFileWriter(const string& root_dir)
    : Processor(PROCESSOR_TYPE_CUSTOM, {"input"}, {}), delay_ms_(0) {
  root_dir_ = root_dir;
  while(root_dir_.back() == '/') {
    root_dir_.pop_back();
  }
}

std::shared_ptr<DBFileWriter> DBFileWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<DBFileWriter>(params.at("filename"));
}

void DBFileWriter::SetRate(int rate) {
  if(rate == 0) {
    delay_ms_ = 0;
  } else {
    delay_ms_ = 1000 / rate;
  }
  expected_timestamp_ = 0;
}

bool DBFileWriter::Init() {
  // Create the root directory if it doesn't exist
  if(!CreateDirs(root_dir_)) {
    LOG(INFO) << "Using existing directory " << root_dir_;
  }
  return true;
}

bool DBFileWriter::OnStop() {
  return true;
}

void DBFileWriter::Process() {
  auto frame = GetFrame("input");
  cv::Mat image = frame->GetValue<cv::Mat>("Image");
  // TODO?: These should probably come from the frame, but not sure what format
  struct tm local_time;
  std::chrono::system_clock::time_point t = std::chrono::system_clock::now();
  time_t now = std::chrono::system_clock::to_time_t(t);
  localtime_r(&now, &local_time);
  const std::chrono::duration<double> tse = t.time_since_epoch();
  
  int sec = local_time.tm_sec;
  int min = local_time.tm_min;
  int hour = local_time.tm_hour;
  int day = local_time.tm_mday;
  int month = local_time.tm_mon + 1;
  int year = local_time.tm_year + 1900;
  int ms = std::chrono::duration_cast<std::chrono::milliseconds>(tse).count() % 1000;
  unsigned long long cur_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tse).count(); 
  // Framerate limiting
  if(expected_timestamp_ == 0) {
    expected_timestamp_ = cur_timestamp;
  }
  if(cur_timestamp < expected_timestamp_) {
    return;
  } else {
    expected_timestamp_ += delay_ms_;
  }
  std::stringstream directory_name;
  directory_name << root_dir_ << "/";
  directory_name << std::setw(4) << std::setfill('0') << year << "-";
  directory_name << std::setw(2) << std::setfill('0') << month << "-";
  directory_name << std::setw(2) << std::setfill('0') << day << "/";
  directory_name << std::setw(2) << std::setfill('0') << hour << "00";

  CreateDirs(directory_name.str());

  std::stringstream filename;
  std::stringstream metadata_filename;
  filename << directory_name.str() << "/";
  filename << std::setw(2) << std::setfill('0') << hour << "_";
  filename << std::setw(2) << std::setfill('0') << min << "-";
  filename << std::setw(2) << std::setfill('0') << sec << "_";
  filename << std::setw(3) << std::setfill('0') << ms;

  metadata_filename << filename.str() << ".json";
  filename << ".png";

  nlohmann::json metadata_json;
  metadata_json["Exposure"] = frame->GetValue<float>("CameraSettings.Exposure");
  metadata_json["Sharpness"] = frame->GetValue<float>("CameraSettings.Sharpness");
  metadata_json["Brightness"] = frame->GetValue<float>("CameraGettings.Brightness");
  metadata_json["Saturation"] = frame->GetValue<float>("CameraSettings.Saturation");
  metadata_json["Hue"] = frame->GetValue<float>("CameraSettings.Hue");
  metadata_json["Gain"] = frame->GetValue<float>("CameraSettings.Gain");
  metadata_json["Gamma"] = frame->GetValue<float>("CameraSettings.Gamma");
  metadata_json["WBRed"] = frame->GetValue<float>("CameraSettings.WBRed");
  metadata_json["WBBlue"] = frame->GetValue<float>("CameraSettings.WBBlue");


  nlohmann::json j;
  j["Metadata"] = metadata_json;
  
  std::string json_string = j.dump();

  // write out metadata
  std::ofstream file;
  file.open(metadata_filename.str());
  file << json_string;
  file.close();

  std::vector<int> params;
  params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  struct stat buf;
  // Write out file
  if(stat (filename.str().c_str(), &buf) != 0) {
    imwrite(filename.str(), image, params);
  }
  // Write to DB
  DoWriteDB(filename.str(), now, frame);
}
