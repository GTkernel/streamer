#include "frame_sender.h"

constexpr auto SOURCE = "input";

FrameSender::FrameSender(std::shared_ptr<grpc::Channel> channel)
    : Processor({SOURCE}, {}), stub_(Messenger::NewStub(channel)) {}

ProcessorType FrameSender::GetType() const {
  return PROCESSOR_TYPE_FRAME_SENDER;
}

void FrameSender::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE, stream);
}

bool FrameSender::Init() { return true; }

bool FrameSender::OnStop() { return true; }

void FrameSender::Process() {
  // TODO:  Support more than just ImageFrame
  auto frame = this->GetFrame<ImageFrame>(SOURCE);
  cv::Mat image = frame->GetImage();

  // serialize
  std::stringstream frame_string;
  try {
    boost::archive::binary_oarchive ar(frame_string);
    ar << image;
  } catch (const boost::archive::archive_exception &e) {
    LOG(INFO) << "Boost serialization error: " << e.what();
  }

  SingleFrame frame_message;
  frame_message.set_frame(frame_string.str());

  grpc::ClientContext context;
  google::protobuf::Empty ignored;
  grpc::Status status = stub_->SendFrame(&context, frame_message, &ignored);

  if (!status.ok()) {
    LOG(INFO) << "gRPC error(SendFrame): " << status.error_message();
  }
}
