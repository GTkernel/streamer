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

#ifndef STREAMER_PROCESSOR_RPC_FRAME_SENDER_H_
#define STREAMER_PROCESSOR_RPC_FRAME_SENDER_H_

#include <grpc++/grpc++.h>

#include "common/types.h"
#include "processor/processor.h"
#include "streamer_rpc.grpc.pb.h"

class FrameSender : public Processor {
 public:
  FrameSender(const std::string server_url);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  static std::shared_ptr<FrameSender> Create(const FactoryParamsType& params);

 protected:
  bool Init() override;
  bool OnStop() override;
  void Process() override;

 private:
  std::string server_url_;
  std::unique_ptr<Messenger::Stub> stub_;
};

#endif  // STREAMER_PROCESSOR_RPC_FRAME_SENDER_H_
