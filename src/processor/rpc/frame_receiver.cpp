#include "frame_receiver.h"

constexpr auto SINK = "output";

FrameReceiver::FrameReceiver(std::string server_url)
    : Processor({}, {SINK}), server_url_(server_url) {
}

void FrameReceiver::RunServer(std::string server_url) {
  grpc::ServerBuilder builder;

  // [NasrinJaleel93] The frame size of live video was exceeding the
  // default value. So I set this too allow that. 6220882 is the size
  // of the live video frame after serialisation and the default size
  // of the message is 4194304.
  builder.SetMaxMessageSize(6320900);
  // TODO:  Use secure credentials (e.g., SslCredentials)
  builder.AddListeningPort(server_url, grpc::InsecureServerCredentials());
  builder.RegisterService(this);

  server_ = builder.BuildAndStart();
  LOG(INFO) << "gRPC server started at " << server_url;
  server_->Wait();
}

grpc::Status FrameReceiver::SendFrame(grpc::ServerContext *context,
                                      const SingleFrame *frame_message,
                                      google::protobuf::Empty *ignored) {
  auto streamname = frame_message->streamname();
  auto frame = frame_message->frame();

  std::string frame_string = frame;
  std::stringstream ss;
  ss.str(frame_string);
  cv::Mat new_mat;

  boost::archive::binary_iarchive lmat(ss);
  lmat >> new_mat;

  PushFrame(SINK, new ImageFrame(new_mat, new_mat));

  return grpc::Status::OK;
}

ProcessorType FrameReceiver::GetType() { return PROCESSOR_TYPE_FRAME_RECEIVER; }

StreamPtr FrameReceiver::GetSink() { return Processor::GetSink(SINK); }

bool FrameReceiver::Init() {
  std::thread server_thread([this]() { this->RunServer(this->server_url_); });
  server_thread.detach();
  return true;
}

bool FrameReceiver::OnStop() {
  server_->Shutdown();
  return true;
}

void FrameReceiver::Process() {
  // Frame is already put in the sink of this processor, no need for
  // process to do anything.
}
