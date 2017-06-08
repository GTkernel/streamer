#include "frame_receiver.h"

constexpr auto SINK = "output";

FrameReceiver::FrameReceiver(const std::string listen_url)
    : Processor({}, {SINK}), listen_url_(listen_url) {}

void FrameReceiver::RunServer(const std::string listen_url) {
  grpc::ServerBuilder builder;

  // TODO:  Use secure credentials (e.g., SslCredentials)
  builder.AddListeningPort(listen_url, grpc::InsecureServerCredentials());
  builder.RegisterService(this);

  server_ = builder.BuildAndStart();
  LOG(INFO) << "gRPC server started at " << listen_url;
  server_->Wait();
}

grpc::Status FrameReceiver::SendFrame(grpc::ServerContext *,
                                      const SingleFrame *frame_message,
                                      google::protobuf::Empty *) {
  std::stringstream frame_string;
  frame_string << frame_message->frame();

  cv::Mat image;

  // If Boost's serialization fails, it throws an exception.  If we don't
  // catch the exception, gRPC triggers a core dump with a "double free or
  // corruption" error. See https://github.com/grpc/grpc/issues/3071
  try {
    boost::archive::binary_iarchive ar(frame_string);
    ar >> image;
  } catch (const boost::archive::archive_exception &e) {
    std::ostringstream error_message;
    error_message << "Boost serialization error: " << e.what();
    LOG(INFO) << error_message.str();
    return grpc::Status(grpc::StatusCode::ABORTED, error_message.str());
  }

  // TODO:  Support more than just ImageFrame
  PushFrame(SINK, new ImageFrame(image, image));

  return grpc::Status::OK;
}

ProcessorType FrameReceiver::GetType() const {
  return PROCESSOR_TYPE_FRAME_RECEIVER;
}

StreamPtr FrameReceiver::GetSink() { return Processor::GetSink(SINK); }

bool FrameReceiver::Init() {
  std::thread server_thread([this]() { this->RunServer(this->listen_url_); });
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
