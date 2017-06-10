
#ifndef STREAMER_PROCESSOR_RPC_FRAME_SENDER_H_
#define STREAMER_PROCESSOR_RPC_FRAME_SENDER_H_

#include <grpc++/grpc++.h>

#include "common/types.h"
#include "processor/processor.h"
#include "processor/rpc/serialization.h"
#include "streamer_rpc.grpc.pb.h"

class FrameSender : public Processor {
 public:
  FrameSender(const std::string server_url);

  void SetSource(StreamPtr stream);
  void SetSource(const string& name, StreamPtr stream) override;

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
