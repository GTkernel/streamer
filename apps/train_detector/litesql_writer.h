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

#ifndef STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_
#define STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_

#include <memory>
#include <string>

#include "litesql.hpp"

#include "common/types.h"
#include "processor/processor.h"

class LiteSqlWriter : public Processor {
 public:
  LiteSqlWriter(const std::string& output_dir);

  static std::shared_ptr<LiteSqlWriter> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string output_dir_;
};

#endif  // STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_
