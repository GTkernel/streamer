
#include "db_filewriter.h"

#include <ctime>
#include <fstream>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <opencv2/opencv.hpp>

#include "json/src/json.hpp"
#include "sys/stat.h"
#include "utils/file_utils.h"

#include "framesdatabase.hpp"

// TODO: this will change to take in the frame after the metadata is added to
// the frame
void DoWriteDB(std::string filename, time_t cur_time,
               std::unique_ptr<Frame>& frame, std::string root_dir) {
  // assumes database has been created
  std::string database_path = "database=" + root_dir + "/frames.db";
  FramesDatabase db("sqlite3", database_path);
  try {
    db.create();
    LOG(INFO) << "Created new database";
  } catch (litesql::Except e) {
    std::cout << e.what();
    LOG(INFO) << "Using existing database";
  }
  try {
    db.begin();
    db.verbose = true;
    FrameEntry fe(db);
    boost::filesystem::path filename_path(filename);
    fe.path = boost::filesystem::canonical(filename_path).string();
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
    db.commit();
  } catch (litesql::Except e) {
    LOG(FATAL) << database_path
               << " doesn't appear to be a valid sqlite3 database.\n"
               << e.what();
  }
}

DBFileWriter::DBFileWriter(const std::string& root_dir)
    : Processor(PROCESSOR_TYPE_CUSTOM, {"input"}, {}) {
  root_dir_ = root_dir;
  while (root_dir_.back() == '/') {
    root_dir_.pop_back();
  }
}

std::shared_ptr<DBFileWriter> DBFileWriter::Create(
    const FactoryParamsType& params) {
  return std::make_shared<DBFileWriter>(params.at("filename"));
}

bool DBFileWriter::Init() {
  // Create the root directory if it doesn't exist
  if (!CreateDirs(root_dir_)) {
    LOG(INFO) << "Using existing directory " << root_dir_;
  }
  return true;
}

bool DBFileWriter::OnStop() { return true; }

void DBFileWriter::Process() {
  auto frame = GetFrame("input");
  cv::Mat image = frame->GetValue<cv::Mat>("original_image");
  auto raw_image = frame->GetValue<std::vector<char>>("compressed_bytes");

  boost::posix_time::ptime pt =
      frame->GetValue<boost::posix_time::ptime>("capture_time_micros");
  tm time_s = boost::posix_time::to_tm(pt);
  int sec = time_s.tm_sec;
  int min = time_s.tm_min;
  int hour = time_s.tm_hour;
  int day = time_s.tm_mday;
  int month = time_s.tm_mon + 1;
  int year = time_s.tm_year + 1900;
  // This is the time offset into the current day.
  boost::posix_time::time_duration time_of_day = pt.time_of_day();
  // We divide by 1000 because the return value of "fractional_seconds()" is in
  // microseconds.
  long ms = time_of_day.fractional_seconds() / 1000;

  std::stringstream directory_name;
  directory_name << root_dir_ << "/";
  directory_name << std::setw(4) << std::setfill('0') << year << "-";
  directory_name << std::setw(2) << std::setfill('0') << month << "-";
  directory_name << std::setw(2) << std::setfill('0') << day << "/";
  directory_name << std::setw(2) << std::setfill('0') << hour << "00";

  CreateDirs(directory_name.str());

  std::stringstream filename;
  filename << directory_name.str() << "/";
  filename << std::setw(2) << std::setfill('0') << hour << "_";
  filename << std::setw(2) << std::setfill('0') << min << "-";
  filename << std::setw(2) << std::setfill('0') << sec << "_";
  filename << std::setw(3) << std::setfill('0') << ms;

  std::stringstream metadata_filename;
  metadata_filename << filename.str() << ".json";

  std::stringstream raw_filename;
  raw_filename << filename.str();

  filename << ".jpg";

  nlohmann::json metadata_json;
  metadata_json["Exposure"] = frame->GetValue<float>("CameraSettings.Exposure");
  metadata_json["Sharpness"] =
      frame->GetValue<float>("CameraSettings.Sharpness");
  metadata_json["Brightness"] =
      frame->GetValue<float>("CameraSettings.Brightness");
  metadata_json["Saturation"] =
      frame->GetValue<float>("CameraSettings.Saturation");
  metadata_json["Hue"] = frame->GetValue<float>("CameraSettings.Hue");
  metadata_json["Gain"] = frame->GetValue<float>("CameraSettings.Gain");
  metadata_json["Gamma"] = frame->GetValue<float>("CameraSettings.Gamma");
  metadata_json["WBRed"] = frame->GetValue<float>("CameraSettings.WBRed");
  metadata_json["WBBlue"] = frame->GetValue<float>("CameraSettings.WBBlue");

  nlohmann::json j;
  j["Metadata"] = metadata_json;

  std::string json_string = j.dump();

  // write out metadata
  std::ofstream metadata_file;
  metadata_file.open(metadata_filename.str());
  metadata_file << json_string;
  metadata_file.close();

  // Write raw file
  std::ofstream raw_file;
  raw_file.open(raw_filename.str(), std::ios::binary | std::ios::out);
  raw_file.write((char*)raw_image.data(), raw_image.size());
  raw_file.close();

  std::vector<int> params;
  /*
  params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  */
  struct stat buf;
  // Write out file
  if (stat(filename.str().c_str(), &buf) != 0) {
    // Rotate 90 degrees clockwise
    cv::Mat dst;
    cv::transpose(image, dst);
    cv::flip(dst, dst, 1);

    imwrite(filename.str(), dst, params);
    std::cout << frame->ToString() << std::endl;
  }

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration time_since_epoch = pt - epoch;
  // This time representation is at the granularity of seconds.
  time_t tt = time_t(time_since_epoch.total_seconds());
  // Write to DB
  DoWriteDB(filename.str(), tt, frame, root_dir_);
}
