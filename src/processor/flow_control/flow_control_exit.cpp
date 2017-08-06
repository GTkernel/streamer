
#include "processor/flow_control/flow_control_exit.h"

#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FlowControlExit::FlowControlExit()
    : Processor(PROCESSOR_TYPE_FLOW_CONTROL_EXIT, {SOURCE_NAME}, {SINK_NAME}) {}

std::shared_ptr<FlowControlExit> FlowControlExit::Create(
    const FactoryParamsType&) {
  return std::make_shared<FlowControlExit>();
}

void FlowControlExit::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr FlowControlExit::GetSink() { return Processor::GetSink(SINK_NAME); }

bool FlowControlExit::Init() { return true; }

bool FlowControlExit::OnStop() { return true; }

void FlowControlExit::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  auto entrance = frame->GetFlowControlEntrance();
  if (entrance) {
    entrance->ReturnToken();
    // Change the frame's FlowControlEntrance to null so that it does not try to
    // release the token again.
    frame->SetFlowControlEntrance(nullptr);
  }
  PushFrame(SINK_NAME, std::move(frame));
}
