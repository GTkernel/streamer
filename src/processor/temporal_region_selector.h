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

#ifndef STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_
#define STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_

#include <string>

#include "common/types.h"
#include "processor/processor.h"

// A processor that selects a region of frames based on their frame id.
class TemporalRegionSelector : public Processor {
 public:
  TemporalRegionSelector(unsigned long start_id, unsigned long end_id);
  static std::shared_ptr<TemporalRegionSelector> Create(
      const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  unsigned long start_id_;
  unsigned long end_id_;
};

#endif  // STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_
