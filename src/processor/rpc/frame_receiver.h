#ifndef STREAMER_FRAME_RECEIVER_H
#define STREAMER_FRAME_RECEIVER_H

#include <grpc++/grpc++.h>

#include "processor/processor.h"
#include "serialization.h"
#include "streamer_rpc.grpc.pb.h"

class FrameReceiver final : public Processor, public Messenger::Service {
 public:
  FrameReceiver(const std::string listen_url);

  ProcessorType GetType() const override;
  StreamPtr GetSink();
  StreamPtr GetSink(const string &name) = delete;

  void RunServer(const std::string listen_url);
  grpc::Status SendFrame(grpc::ServerContext *context,
                         const SingleFrame *frame_message,
                         google::protobuf::Empty *ignored) override;

 protected:
  bool Init() override;
  bool OnStop() override;
  void Process() override;

 private:
  std::string listen_url_;
  std::unique_ptr<grpc::Server> server_;
};

#endif  // STREAMER_FRAME_RECEIVER_H
