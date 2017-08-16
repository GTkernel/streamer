
#include "processor/strider.h"
#include "processor/flow_control/flow_control_entrance.h"

#include <stdexcept>

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Strider::Strider(unsigned int stride)
    : Processor(PROCESSOR_TYPE_STRIDER, {SOURCE_NAME}, {SINK_NAME}),
      stride_(stride) {}

std::shared_ptr<Strider> Strider::Create(const FactoryParamsType& params) {
  int stride = std::stoi(params.at("stride"));
  if (stride < 0) {
    throw std::invalid_argument("\"stride\" cannot be negative, but is: " +
                                std::to_string(stride));
  }
  return std::make_shared<Strider>((unsigned int)stride);
}

void Strider::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Strider::GetSink() { return Processor::GetSink(SINK_NAME); }

bool Strider::Init() { return true; }

bool Strider::OnStop() { return true; }

void Strider::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  ++num_frames_processed_;

  if (num_frames_processed_ % stride_ == 0) {
    PushFrame(SINK_NAME, std::move(frame));
  } else {
    // Drop frame
    unsigned long id = frame->GetValue<unsigned long>("frame_id");
    LOG(WARNING) << "Dropping frame " << id << " in strider.";
    auto flow_control_entrance = frame->GetFlowControlEntrance();
    if (flow_control_entrance) {
      // If a flow control entrance exists, then we need to inform it that a
      // frame is being dropped so that the flow control token is returned.
      flow_control_entrance->ReturnToken();
      // Change the frame's FlowControlEntrance to null so that it does not try
      // to release the token again.
      frame->SetFlowControlEntrance(nullptr);
    }
  }
}
