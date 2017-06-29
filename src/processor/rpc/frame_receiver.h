
#ifndef STREAMER_PROCESSOR_RPC_FRAME_RECEIVER_H_
#define STREAMER_PROCESSOR_RPC_FRAME_RECEIVER_H_

#include <grpc++/grpc++.h>

#include "common/types.h"
#include "processor/processor.h"
#include "streamer_rpc.grpc.pb.h"

class FrameReceiver final : public Processor, public Messenger::Service {
 public:
  FrameReceiver(const std::string listen_url);

  StreamPtr GetSink();
  StreamPtr GetSink(const string& name) = delete;

  void RunServer(const std::string listen_url);
  grpc::Status SendFrame(grpc::ServerContext* context,
                         const SingleFrame* frame_message,
                         google::protobuf::Empty* ignored) override;

  static std::shared_ptr<FrameReceiver> Create(const FactoryParamsType& params);

 protected:
  bool Init() override;
  bool OnStop() override;
  void Process() override;

 private:
  std::string listen_url_;
  std::unique_ptr<grpc::Server> server_;
};

#endif  // STREAMER_PROCESSOR_RPC_FRAME_RECEIVER_H_
