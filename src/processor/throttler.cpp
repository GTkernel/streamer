
#include "processor/throttler.h"

#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "streamer.h"

Throttler::Throttler(int fps)
    : Processor(PROCESSOR_TYPE_THROTTLER, {"input"}, {"output"}), fps_(fps) {}

std::shared_ptr<Throttler> Throttler::Create(const FactoryParamsType& params) {
  int fps = std::stoi(params.at("fps"));
  return std::make_shared<Throttler>(fps);
}

bool Throttler::Init() {
  timer_.Start();
  SetFps(fps_);
  return true;
}

bool Throttler::OnStop() { return true; }

void Throttler::Process() {
  auto frame = GetFrame("input");
  double elapsed_ms = timer_.ElapsedMSec();
  if (elapsed_ms < delay_ms_) {
    // Drop frame
    auto flow_control_entrance = frame->GetFlowControlEntrance();
    if (flow_control_entrance) {
      // If a flow control entrance exists, then we need to inform it that a
      // frame is being dropped so that the flow control token is returned.
      flow_control_entrance->ReturnToken();
      // Change the frame's FlowControlEntrance to null so that it does not try
      // to release the token again.
      frame->SetFlowControlEntrance(nullptr);
    }
    return;
  } else {
    LOG(WARNING) << "Frame rate too high. Dropping frame: "
                 << frame->GetValue<unsigned long>("frame_id");
  }

  // Restart timer
  timer_.Start();
  PushFrame("output", std::move(frame));
}

void Throttler::SetFps(int fps) {
  if (fps == 0) {
    LOG(WARNING) << "Tried to set FPS to zero.";
    return;
  } else if (fps == -1) {
    // Turn throttling off with -1
    delay_ms_ = 0;
  } else {
    delay_ms_ = 1000 / fps;
  }
}
