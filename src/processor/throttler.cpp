
#include "throttler.h"
#include "model/model_manager.h"
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
    return;
  }
  // Restart timer
  timer_.Start();

  // Make new frame to pass through, conditioned on its type
  // TODO: Do we need to actually make a new copy of the frame? - Thomas
  Frame* output_frame = new Frame();
  switch (frame->GetType()) {
    case FRAME_TYPE_LAYER: {
      output_frame->SetOriginalImage(frame->GetOriginalImage());
      output_frame->SetActivations(frame->GetActivations());
      output_frame->SetLayerName(frame->GetLayerName());
      break;
    }
    case FRAME_TYPE_IMAGE: {
      output_frame->SetOriginalImage(frame->GetOriginalImage());
      output_frame->SetImage(frame->GetImage());
      break;
    }
    case FRAME_TYPE_BYTES: {
      output_frame->SetOriginalImage(frame->GetOriginalImage());
      output_frame->SetDataBuffer(frame->GetDataBuffer());
      break;
    }
    case FRAME_TYPE_MD: {
      output_frame->SetOriginalImage(frame->GetOriginalImage());
      output_frame->SetTags(frame->GetTags());
      output_frame->SetBboxes(frame->GetBboxes());
      break;
    }
    default: { STREAMER_NOT_IMPLEMENTED; }
  }
  PushFrame("output", output_frame);
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
