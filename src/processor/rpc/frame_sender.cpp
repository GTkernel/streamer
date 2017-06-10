#include "frame_sender.h"

#include "common/types.h"

constexpr auto SOURCE = "input";

FrameSender::FrameSender(const std::string server_url)
    : Processor(PROCESSOR_TYPE_FRAME_SENDER, {SOURCE}, {}),
      server_url_(server_url) {
  auto channel =
      grpc::CreateChannel(server_url_, grpc::InsecureChannelCredentials());
  stub_ = Messenger::NewStub(channel);
}

void FrameSender::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE, stream);
}

void FrameSender::SetSource(const string& name, StreamPtr stream) {
  CHECK(name == SOURCE) << "StreamSender has one source named \"" << SOURCE
                        << "\"!";
  Processor::SetSource(name, stream);
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
  } catch (const boost::archive::archive_exception& e) {
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
