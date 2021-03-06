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

#ifndef STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_
#define STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_

#include <memory>

#include "common/types.h"
#include "processor/processor.h"

// FlowControlExit is used to update the token count maintained by
// FlowControlEntrance when frames leave the pipeline. Both classes together are
// used to limit the number of outstanding frames in the pipeline.
class FlowControlExit : public Processor {
 public:
  FlowControlExit();
  static std::shared_ptr<FlowControlExit> Create(
      const FactoryParamsType& params);

  void SetSink(StreamPtr stream);
  using Processor::SetSink;

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;
};

#endif  // STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_
