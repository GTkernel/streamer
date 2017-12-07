
#include "processor/buffer.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Buffer::Buffer(unsigned long num_frames)
    : Processor(PROCESSOR_TYPE_THROTTLER, {SOURCE_NAME}, {SINK_NAME}),
      buffer_{num_frames} {}

std::shared_ptr<Buffer> Buffer::Create(const FactoryParamsType& params) {
  return std::make_shared<Buffer>(std::stoi(params.at("num_frames")));
}

void Buffer::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Buffer::GetSink() { return Processor::GetSink(SINK_NAME); }

bool Buffer::Init() { return true; }

bool Buffer::OnStop() { return true; }

void Buffer::Process() {
  if (buffer_.full()) {
    PushFrame(SINK_NAME, std::move(buffer_.front()));
    buffer_.pop_front();
  }

  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  buffer_.push_back(std::move(frame));
}
