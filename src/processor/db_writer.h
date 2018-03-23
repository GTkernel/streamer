// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Send metadata to the database
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_DB_WRITER_H_
#define STREAMER_PROCESSOR_DB_WRITER_H_

#ifdef USE_ATHENA
#include "client/AthenaClient.h"
#endif  // USE_ATHENA
#include "camera/camera.h"
#include "processor.h"

class DbWriter : public Processor {
 public:
  DbWriter(std::shared_ptr<Camera> camera, bool write_to_file,
           const std::string& athena_address);
  static std::shared_ptr<DbWriter> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  void WriteFile(const std::string& camera_id,
                 const std::vector<std::string>& uuids, unsigned long timestamp,
                 const std::vector<std::string>& tags,
                 const std::vector<std::vector<double>>& features);
  void WriteAthena(const std::string& camera_id,
                   const std::vector<std::string>& uuids,
                   unsigned long timestamp,
                   const std::vector<std::string>& tags,
                   const std::vector<std::vector<double>>& features);

 private:
  std::shared_ptr<Camera> camera_;
  bool write_to_file_;
  std::ofstream ofs_;
  std::string athena_address_;
#ifdef USE_ATHENA
  std::unique_ptr<athena::AthenaClient> aclient_;
#endif  // USE_ATHENA
};

#endif  // STREAMER_PROCESSOR_DB_WRITER_H_
