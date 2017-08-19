/**
 * Send metadata to the database
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>

#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/spirit/include/qi.hpp"
#include "boost/tokenizer.hpp"

#include "db_writer.h"

DbWriter::DbWriter(std::shared_ptr<Camera> camera, bool write_to_file,
                   const std::string& athena_address)
    : Processor(PROCESSOR_TYPE_DB_WRITER, {"input"}, {}),
      camera_(camera),
      write_to_file_(write_to_file),
      athena_address_(athena_address) {}

std::shared_ptr<DbWriter> DbWriter::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool DbWriter::Init() {
  if (write_to_file_) {
    char tplate[7] = "XXXXXX";
    if (mkstemp(tplate) == -1) {
      throw std::runtime_error("Unable to create temporary file!");
    }
    ofs_.open(tplate);
    CHECK(ofs_.is_open()) << "Error opening file";
  } else {
    if (!athena_address_.empty()) {
#ifdef USE_ATHENA
      aclient_.reset(new athena::AthenaClient(athena_address_));
#else
      LOG(FATAL)
          << "Athena client not supported, please compile with -DUSE_ATHENA=ON";
#endif  // USE_ATHENA
    }
  }
  return true;
}

bool DbWriter::OnStop() {
  if (write_to_file_) {
    ofs_.close();
  }
  return true;
}

static unsigned long GetTimeSinceEpochMillis() {
  return static_cast<unsigned long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

void DbWriter::Process() {
  auto frame = GetFrame("input");
  auto camera_id = camera_->GetName();
  auto uuids = frame->GetValue<std::vector<std::string>>("uuids");
  auto timestamp = GetTimeSinceEpochMillis();
  auto tags = frame->GetValue<std::vector<std::string>>("tags");
  auto features =
      frame->GetValue<std::vector<std::vector<double>>>("features");
  CHECK(uuids.size() == tags.size());
  if (write_to_file_) {
    WriteFile(camera_id, uuids, timestamp, tags, features);
  } else {
#ifdef USE_ATHENA
    if (!athena_address_.empty() && aclient_)
      WriteAthena(camera_id, uuids, timestamp, tags, features);
#endif  // USE_ATHENA
  }
}

void DbWriter::WriteFile(
    const std::string& camera_id, const std::vector<std::string>& uuids,
    unsigned long timestamp, const std::vector<string>& tags,
    const std::vector<std::vector<double>>& features) {
  for (size_t i = 0; i < uuids.size(); ++i) {
    if (uuids.size() == features.size()) {
      ofs_ << camera_id << "," << uuids[i] << "," << timestamp << ","
           << tags[i];
      bool flag = true;
      for (const auto& m : features[i]) {
        if (flag) {
          ofs_ << "," << m;
          flag = false;
        } else {
          ofs_ << ";" << m;
        }
      }
      ofs_ << "\n";
    } else {
      ofs_ << camera_id << "," << uuids[i] << "," << timestamp << ","
           << tags[i];
      ofs_ << "\n";
    }
  }
}

#ifdef USE_ATHENA
void DbWriter::WriteAthena(
    const std::string& camera_id, const std::vector<std::string>& uuids,
    unsigned long timestamp, const std::vector<string>& tags,
    const std::vector<std::vector<double>>& features) {
  for (size_t i = 0; i < uuids.size(); ++i) {
    if (uuids.size() == features.size()) {
      const std::string& streamId = camera_id;
      const std::string& objectId = uuids[i];
      const std::vector<double>& fv = features[i];
      // Turn the metadata into a JSON object

      using boost::property_tree::ptree;

      ptree root;

      root.put("id", objectId);
      root.put("ts", timestamp);
      root.put("st", streamId);

      ptree tag_array;
      ptree fv_array;

      tag_array.push_back(std::make_pair("", ptree(tags[i])));
      std::for_each(fv.begin(), fv.end(), [&](const double& v) {
        fv_array.push_back(std::make_pair("", ptree(std::to_string(v))));
      });

      root.put_child("tg", tag_array);
      root.put_child("fv", fv_array);

      std::ostringstream oss;
      boost::property_tree::write_json(oss, root, false);

      // The following is for JSON compliance

      std::string json_str = std::regex_replace(
          oss.str(), std::regex(R"sy("([-+]?[0-9]*\.?[0-9]*)")sy"), "$1");

      std::cout << json_str << std::endl;

      std::cout << aclient_->query(json_str) << std::endl;
    }
  }
}
#endif  // USE_ATHENA
