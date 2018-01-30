
#include "fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FvGen::FvGen()
    : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}) {
}

FvGen::~FvGen() {}

void FvGen::AddFv(std::string layer_name, int xmin, int xmax, int ymin, int ymax, bool flat) {
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

void FvGen::SetSource(StreamPtr stream) {
  SetSource(SOURCE_NAME, stream);
}

StreamPtr FvGen::GetSink() { return Processor::GetSink(SINK_NAME); }

void FvGen::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  for(auto& spec : feature_vector_specs_) {
    cv::Mat input_mat = input_frame->GetValue<cv::Mat>(spec.layer_name_);
    cv::Mat fv;
    cv::Mat new_fv;
    if(spec.roi_.height != 0 && spec.roi_.width != 0) {
      fv = cv::Mat(input_mat.rows, input_mat.cols, CV_32FC(input_mat.channels()));
      fv = input_mat(spec.roi_);
    }
    else {
      fv = input_mat;
    }
    if(spec.flat_) {
      new_fv = cv::Mat(fv.rows * fv.cols * fv.channels(), 1, 1, fv.clone().data);
      LOG(INFO) << new_fv.rows << " "<< new_fv.cols << " "<< new_fv.channels();
    }
    input_frame->SetValue(FvSpec::GetUniqueID(spec), fv);
  }
  PushFrame(SINK_NAME, std::move(input_frame));
}
