
#include "fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FVGen::FVGen(int xmin, int xmax, int ymin, int ymax)
    : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      crop_roi_(xmin, ymin, xmax - xmin, ymax - ymin) {
    CHECK(xmin < xmax && ymin < ymax) << "Cannot have negative dimensions on crop window";
}

FVGen::~FVGen() {}

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
