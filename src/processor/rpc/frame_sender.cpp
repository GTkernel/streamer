#include "frame_sender.h"

constexpr auto SOURCE = "input";
// TODO: let user set this in constructor
constexpr auto STREAM_NAME = "video_stream";

FrameSender::FrameSender(std::shared_ptr<grpc::Channel> channel)
    : Processor({SOURCE}, {}), stub_(Messenger::NewStub(channel)) {}

ProcessorType FrameSender::GetType() { return PROCESSOR_TYPE_FRAME_SENDER; }

void FrameSender::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE, stream);
}

bool FrameSender::Init() { return true; }

bool FrameSender::OnStop() { return true; }

void FrameSender::Process() {
  // TODO:  Support more than just ImageFrame
  auto frame = this->GetFrame<ImageFrame>(SOURCE);
  cv::Mat img = frame->GetImage();

  // serialize
  std::stringstream frame_string;
  binary_oarchive oa{frame_string};
  oa << img;

  SingleFrame frame_message;
  frame_message.set_frame(frame_string.str());
  frame_message.set_streamname(STREAM_NAME);

  grpc::ClientContext context;
  google::protobuf::Empty ignored;
  grpc::Status status = stub_->SendFrame(&context, frame_message, &ignored);

  if (!status.ok()) {
    LOG(INFO) << status.error_code() << ": " << status.error_message();
  }
}
