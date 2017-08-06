
#include "processor/flow_control/flow_control_entrance.h"

#include <stdexcept>

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FlowControlEntrance::FlowControlEntrance(unsigned int max_tokens)
    : Processor(PROCESSOR_TYPE_FLOW_CONTROL_ENTRANCE, {SOURCE_NAME},
                {SINK_NAME}),
      max_tokens_(max_tokens),
      num_tokens_available_(max_tokens) {}

std::shared_ptr<FlowControlEntrance> FlowControlEntrance::Create(
    const FactoryParamsType& params) {
  int max_tokens = std::stoi(params.at("max_tokens"));
  if (max_tokens < 0) {
    throw std::invalid_argument("\"max_tokens\" cannot be negative, but is: " +
                                std::to_string(max_tokens));
  }
  return std::make_shared<FlowControlEntrance>((unsigned int)max_tokens);
}

void FlowControlEntrance::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr FlowControlEntrance::GetSink() {
  return Processor::GetSink(SINK_NAME);
}

bool FlowControlEntrance::Init() { return true; }

bool FlowControlEntrance::OnStop() { return true; }

void FlowControlEntrance::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  unsigned long id = frame->GetValue<unsigned long>("frame_id");
  if (frame->GetFlowControlEntrance()) {
    throw std::runtime_error("Frame " + std::to_string(id) +
                             " is already under flow control.");
  }

  // Used to minimize the length of the critical section.
  bool push = false;
  {
    std::lock_guard<std::mutex> guard(mtx_);
    if (num_tokens_available_) {
      --num_tokens_available_;
      push = true;
    }
  }

  if (push) {
    frame->SetFlowControlEntrance(this);
    PushFrame(SINK_NAME, std::move(frame));
  } else {
    LOG(WARNING) << "Insufficient flow control tokens. Dropping frame: " << id;
  }
}

void FlowControlEntrance::ReturnToken() {
  std::lock_guard<std::mutex> guard(mtx_);
  ++num_tokens_available_;
  if (num_tokens_available_ > max_tokens_) {
    throw std::runtime_error(
        "More flow control tokens have been returned than were distributed.");
  }
}