/**
* Send metadata to the database
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#include <chrono>
#include "db_writer.h"

DbWriter::DbWriter(std::shared_ptr<Camera> camera) 
    : Processor({"input"}, {}),
      camera_(camera) {}

bool DbWriter::Init() {
  return true;
}

bool DbWriter::OnStop() {
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
  for (size_t i = 0; i < uuids.size(); ++i) {
    if (uuids.size() == struck_features.size())
      LOG(INFO) << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i] << "[struck_feature]";
    else
      LOG(INFO) << camera_id << "," << uuids[i] << "," << timestamp << "," << tags[i];
  }
}

ProcessorType DbWriter::GetType() {
  return PROCESSOR_TYPE_DB_WRITER;
}
