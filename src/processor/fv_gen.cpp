
#include "fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FvGen::FvGen(int xmin, int ymin, int xmax, int ymax, bool flat)
    : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}) {
    CHECK(xmin < xmax && ymin < ymax) << "Cannot have negative dimensions on crop window";
}

FvGen::~FvGen() {}

void FvGen::AddFV(std::string layer_name, int xmin, int xmax, int ymin, int ymax, bool flat) {
  feature_vector_specs_.push_back(FvSpec(layer_name, xmin, xmax, ymin, ymax, flat));

}
std::string FvSpec::GetUniqueID(const FvSpec& spec) {
  std::ostringstream ss;
  ss << spec.layer_name_ << spec.xmin_ << spec.ymin_ << spec.xmax_ << spec.ymax_ << std::boolalpha << spec.flat_;
  return ss.str();
}

std::shared_ptr<FvGen> FvGen::Create(
    const FactoryParamsType& params) {
  (void) params;
  return nullptr;
}

bool FvGen::Init() { return true; }

bool FvGen::OnStop() { return true; }

void FvGen::SetSource(const std::string& name, StreamPtr stream) {
  Processor::SetSource(name, stream);
}

void FvGen::SetSource(StreamPtr stream) {
  SetSource(SOURCE_NAME, stream);
}

void FvGen::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  cv::Mat input_mat = input_frame->GetValue<cv::Mat>("activations");
  for(auto& spec : feature_vector_specs_) {
    cv::Mat fv = input_mat(spec.roi_);
    if(spec.flat_) {
      fv = fv.reshape(fv.rows * fv.cols * fv.channels());
    }
    input_frame->SetValue(spec.layer_name_, fv);
  }
  PushFrame(SINK_NAME, std::move(input_frame));
}
