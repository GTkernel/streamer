/**
* Send metadata to the database
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#include <chrono>
#include "db_writer.h"

DbWriter::DbWriter(std::shared_ptr<Camera> camera, bool write_to_file) 
    : Processor({"input"}, {}),
      camera_(camera),
      write_to_file_(write_to_file) {}

bool DbWriter::Init() {
  if (write_to_file_) {
    ofs_.open("data_sample.csv");
    CHECK(ofs_.is_open()) << "Error opening file";
  }
  return true;
}

bool DbWriter::OnStop() {
  if (write_to_file_) {
    ofs_.close();
  }
  return true;
}

static unsigned long GetTimeSinceEpochMillis()
{
    return static_cast<unsigned long>
        (std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now().time_since_epoch()).count());
}

void DbWriter::Process() {
  auto md_frame = GetFrame<MetadataFrame>("input");
  auto camera_id = camera_->GetName();
  auto uuids = md_frame->GetUuids();
  auto timestamp = GetTimeSinceEpochMillis();
  auto tags = md_frame->GetTags();
  auto struck_features = md_frame->GetStruckFeatures();
  //auto bboxes = md_frame->GetBboxes();
  CHECK(uuids.size() == tags.size());
  if (write_to_file_)
    WriteFile(camera_id, uuids, timestamp, tags, struck_features);
  else
    WriteAthena(camera_id, uuids, timestamp, tags, struck_features);
}

ProcessorType DbWriter::GetType() {
  return PROCESSOR_TYPE_DB_WRITER;
}

void DbWriter::WriteFile(const std::string& camera_id,
                         const std::vector<std::string>& uuids,
                         unsigned long timestamp,
                         const std::vector<string>& tags,
                         const std::vector<std::vector<double>>& struck_features) {
  for (size_t i = 0; i < uuids.size(); ++i) {
    if (uuids.size() == struck_features.size()) {
      ofs_ << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i];
      bool flag = true;
      for (const auto& m: struck_features[i]) {
        if (flag) {
          ofs_ << "," << m;
          flag = false;
        } else {
          ofs_ << ";" << m;
        }
      }
      ofs_ << "\n";
    } else {
      ofs_ << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i];
      ofs_ << "\n";
    }
  }
}

void DbWriter::WriteAthena(const std::string& camera_id,
                           const std::vector<std::string>& uuids,
                           unsigned long timestamp,
                           const std::vector<string>& tags,
                           const std::vector<std::vector<double>>& struck_features) {
  /*
  for (size_t i = 0; i < uuids.size(); ++i) {
    if (uuids.size() == struck_features.size())
      LOG(INFO) << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i] << "[struck_feature]";
    else
      LOG(INFO) << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i];
  }
  */

  // Waiting for Athena API
}