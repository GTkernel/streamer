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

#ifndef STREAMER_PROCESSOR_FV_GEN_H_
#define STREAMER_PROCESSOR_FV_GEN_H_

#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

class FvSpec {
 public:
  FvSpec() {}
  FvSpec(const std::string& layer_name, int xmin = 0, int ymin = 0,
         int xmax = 0, int ymax = 0, bool flat = true);

  static std::string GetUniqueID(const FvSpec& spec);

 public:
  std::string layer_name_;
  cv::Rect roi_;
  cv::Range yrange_;
  cv::Range xrange_;
  int xmin_, xmax_, ymin_, ymax_;
  bool flat_;
};

class FvGen : public Processor {
 public:
  FvGen();

  static std::shared_ptr<FvGen> Create(const FactoryParamsType& params);

  void AddFv(const std::string& layer_name, int xmin = 0, int ymin = 0,
             int xmax = 0, int ymax = 0, bool flat = false);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::vector<FvSpec> feature_vector_specs_;
};

#endif  // STREAMER_PROCESSOR_FV_GEN_H_
