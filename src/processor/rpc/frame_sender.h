#ifndef STREAMER_FRAME_SENDER_H
#define STREAMER_FRAME_SENDER_H

#include <grpc++/grpc++.h>

#include "processor/processor.h"
#include "serialization.h"
#include "streamer_rpc.grpc.pb.h"

class FrameSender : public Processor {
 public:
  FrameSender(const std::string server_url);

  void SetSource(StreamPtr stream);
  void SetSource(const string& name, StreamPtr stream) override;

 protected:
  bool Init() override;
  bool OnStop() override;
  void Process() override;

 private:
  std::string server_url_;
  std::unique_ptr<Messenger::Stub> stub_;
};

#endif  // STREAMER_FRAME_SENDER_H
