
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

void FlowControlExit::SetSink(StreamPtr stream) {
  Processor::SetSink(SINK_NAME, stream);
}

void FlowControlExit::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr FlowControlExit::GetSink() { return Processor::GetSink(SINK_NAME); }

bool FlowControlExit::Init() { return true; }

bool FlowControlExit::OnStop() { return true; }

void FlowControlExit::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  auto start_time = boost::posix_time::microsec_clock::local_time();
  auto entrance = frame->GetFlowControlEntrance();
  if (entrance) {
    entrance->ReturnToken(frame->GetValue<unsigned long>("frame_id"));
    // Change the frame's FlowControlEntrance to null so that it does not try to
    // release the token again.
    auto end_time = boost::posix_time::microsec_clock::local_time();
    frame->SetValue("flow_control_exit.enter_time", start_time);
    frame->SetValue("flow_control_exit.exit_time", end_time);
    frame->SetFlowControlEntrance(nullptr);
  }
  PushFrame(SINK_NAME, std::move(frame));
}
