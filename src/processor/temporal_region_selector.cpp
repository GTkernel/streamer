
#include "processor/temporal_region_selector.h"

#include "processor/flow_control/flow_control_entrance.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

TemporalRegionSelector::TemporalRegionSelector(unsigned long start_id,
                                               unsigned long end_id)
    : Processor(PROCESSOR_TYPE_TEMPORAL_REGION_SELECTOR, {SOURCE_NAME},
                {SINK_NAME}),
      start_id_(start_id),
      end_id_(end_id) {
  CHECK(end_id_ >= start_id_)
      << "End frame id must be greater than or equal to start frame id.";
}

std::shared_ptr<TemporalRegionSelector> TemporalRegionSelector::Create(
    const FactoryParamsType& params) {
  unsigned long start_id = std::stoul(params.at("start_id"));
  unsigned long end_id = std::stoul(params.at("end_if"));
  return std::make_shared<TemporalRegionSelector>(start_id, end_id);
}

void TemporalRegionSelector::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr TemporalRegionSelector::GetSink() {
  return Processor::GetSink(SINK_NAME);
}

bool TemporalRegionSelector::Init() { return true; }

bool TemporalRegionSelector::OnStop() { return true; }

void TemporalRegionSelector::Process() {
  auto frame = GetFrame("input");

  auto frame_id = frame->GetValue<unsigned long>("frame_id");
  if (frame_id < start_id_) {
    LOG(WARNING) << "Frame " << frame_id << " not in region [" << start_id_
                 << ", " << end_id_ << "]. Dropping frame: " << frame_id;
    // Drop frame
    auto flow_control_entrance = frame->GetFlowControlEntrance();
    if (flow_control_entrance) {
      // If a flow control entrance exists, then we need to inform it that a
      // frame is being dropped so that the flow control token is returned.
      flow_control_entrance->ReturnToken(frame_id);
      // Change the frame's FlowControlEntrance to null so that it does not try
      // to release the token again.
      frame->SetFlowControlEntrance(nullptr);
    }
    return;
  } else if (frame_id > end_id_) {
    LOG(INFO) << __FILE__ << " line: " << __LINE__ << " make_unique";
    auto stop_frame = std::make_unique<Frame>();
    stop_frame->SetStopFrame(true);
    PushFrame(SINK_NAME, std::move(stop_frame));
    return;
  }

  PushFrame(SINK_NAME, std::move(frame));
}
