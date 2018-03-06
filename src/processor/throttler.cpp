
#include "processor/throttler.h"

#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Throttler::Throttler(double fps)
    : Processor(PROCESSOR_TYPE_THROTTLER, {SOURCE_NAME}, {SINK_NAME}),
      delay_ms_(0) {
  SetFps(fps);
}

std::shared_ptr<Throttler> Throttler::Create(const FactoryParamsType& params) {
  double fps = std::stod(params.at("fps"));
  return std::make_shared<Throttler>(fps);
}

void Throttler::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Throttler::GetSink() { return Processor::GetSink(SINK_NAME); }

void Throttler::SetFps(double fps) {
  if (fps < 0) {
    throw std::invalid_argument("Fps cannot be negative!");
  } else if (fps == 0) {
    // Turn throttling off.
    delay_ms_ = 0;
  } else {
    delay_ms_ = 1000 / fps;
  }
}

bool Throttler::Init() { return true; }

bool Throttler::OnStop() { return true; }

void Throttler::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  auto start_time = boost::posix_time::microsec_clock::local_time();

  if (timer_.ElapsedMSec() < delay_ms_) {
    // Drop frame.
    LOG(INFO) << "Frame rate too high. Dropping frame: "
              << frame->GetValue<unsigned long>("frame_id");

    FlowControlEntrance* flow_control_entrance =
        frame->GetFlowControlEntrance();
    if (flow_control_entrance) {
      // If a flow control entrance exists, then we need to inform it that a
      // frame is being dropped so that the flow control token is returned.
      flow_control_entrance->ReturnToken(
          frame->GetValue<unsigned long>("frame_id"));
      // Change the frame's FlowControlEntrance to null so that it does not try
      // to release the token again.
      frame->SetFlowControlEntrance(nullptr);
    }
  } else {
    // Restart timer
    timer_.Start();
    PushFrame(SINK_NAME, std::move(frame));
  }
}
