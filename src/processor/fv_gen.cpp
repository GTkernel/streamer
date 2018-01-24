
#include "fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FVGen::FVGen(const Shape& crop_shape, int x, int y)
    : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      x_(x),
      y_(y),
      crop_roi_(x, y, crop_shape.width, crop_shape.height) {
  // Create sinks.
}

FVGen::~FVGen() {
}

std::shared_ptr<FVGen> FVGen::Create(
    const FactoryParamsType& params) {
  (void) params;
  return nullptr;
}

bool FVGen::Init() { return true; }

bool FVGen::OnStop() { return true; }

void FVGen::SetSource(const std::string& name, StreamPtr stream) {
  Processor::SetSource(name, stream);
}

void FVGen::SetSource(StreamPtr stream) {
  SetSource(SOURCE_NAME, stream);
}

void FVGen::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  cv::Mat input_mat = input_frame->GetValue<cv::Mat>("activations");
  cv::Mat fv = input_mat(crop_roi_);
  input_frame->SetValue("feature_vector", fv);
  PushFrame(SINK_NAME, std::move(input_frame));
}
