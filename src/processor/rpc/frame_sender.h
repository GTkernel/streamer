#ifndef STREAMER_FRAME_SENDER_H
#define STREAMER_FRAME_SENDER_H

#include <grpc++/grpc++.h>

#include "processor/processor.h"
#include "serialization.h"
#include "streamer_rpc.grpc.pb.h"

class FrameSender : public Processor {
 public:
  FrameSender(std::shared_ptr<grpc::Channel> channel);

  ProcessorType GetType() override;
  void SetSource(StreamPtr stream);
  void SetSource(const string &name, StreamPtr stream) = delete;

 protected:
  bool Init() override;
  bool OnStop() override;
  void Process() override;

 private:
  std::unique_ptr<Messenger::Stub> stub_;
};

#endif  // STREAMER_FRAME_SENDER_H
