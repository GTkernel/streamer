
#include "fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FvGen::FvGen() : Processor(PROCESSOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}) {}

FvGen::~FvGen() {}

void FvGen::AddFv(std::string layer_name, int xmin, int xmax, int ymin,
                  int ymax, bool flat) {
  feature_vector_specs_.push_back(
      FvSpec(layer_name, xmin, xmax, ymin, ymax, flat));
}
std::string FvSpec::GetUniqueID(const FvSpec& spec) {
  std::ostringstream ss;
  ss << spec.layer_name_ << spec.xmin_ << spec.ymin_ << spec.xmax_ << spec.ymax_
     << std::boolalpha << spec.flat_;
  return ss.str();
}
std::shared_ptr<FvGen> FvGen::Create(const FactoryParamsType& params) {
  (void)params;
  return nullptr;
}

bool FvGen::Init() { return true; }

bool FvGen::OnStop() { return true; }

void FvGen::SetSource(StreamPtr stream) { SetSource(SOURCE_NAME, stream); }

StreamPtr FvGen::GetSink() { return Processor::GetSink(SINK_NAME); }

void FvGen::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  auto start_time = boost::posix_time::microsec_clock::local_time();

  for (auto& spec : feature_vector_specs_) {
    cv::Mat input_mat = input_frame->GetValue<cv::Mat>(spec.layer_name_);
    cv::Mat fv;
    cv::Mat new_fv;
    if (spec.roi_.height != 0 && spec.roi_.width != 0) {
      // LOG(INFO) << spec.xmin_ << " " << spec.xmax_ << " " << spec.ymin_ << "
      // " << spec.ymax_;
      // LOG(INFO) << input_mat.rows << " " << input_mat.cols << " " <<
      // input_mat.channels(); LOG(INFO) << spec.roi_.x << " " <<
      // spec.roi_.width << " " << spec.roi_.y << " " << spec.roi_.height;
      // LOG(INFO) << spec.yrange_.start << " " << spec.yrange_.end;
      // LOG(INFO) << spec.xrange_.start << " " << spec.xrange_.end;
      fv = input_mat({spec.yrange_, spec.xrange_});
      // LOG(INFO) << fv.rows << " " << fv.cols << " " << fv.channels();
      std::vector<cv::Mat> channels;
#undef DOCHECK
#ifdef DOCHECK
      int full_height = input_mat.size[0];
      int full_width = input_mat.size[1];
      int roi_height = spec.roi_.height;
      int roi_width = spec.roi_.width;
      int channels = input_mat.channels();
      LOG(INFO) << full_height << " " << full_width << " " << channels;
      for (int c = 0; c < channels; ++c) {
        for (int y = spec.ymin_; y < spec.ymax_; ++y) {
          for (int x = spec.xmin_; x < spec.xmax_; ++x) {
            int cropped_y = y - spec.ymin_;
            int cropped_x = x - spec.xmin_;
            LOG(INFO) << "(" << x << ", " << y << ", " << c << ") "
                      << "(" << cropped_x << ", " << cropped_y << ", " << c
                      << ")";
            float lhs = fv.ptr<float>(cropped_y)[channels * cropped_x + c];
            float rhs = input_mat.ptr<float>(y)[channels * x + c];
            LOG(INFO) << lhs << " " << rhs;
            CHECK(lhs == rhs);
          }
        }
      }
#endif
#undef DOCHECK
    } else {
      fv = input_mat;
    }
    if (spec.flat_) {
      // Suspected heap corruption occurring somewhere causing possible segfault
      // in this block
      new_fv =
          cv::Mat(spec.roi_.height * spec.roi_.width * input_mat.channels(), 1,
                  CV_32FC1);
      memcpy(new_fv.data, fv.clone().data,
             spec.roi_.height * spec.roi_.width * input_mat.channels() * 0 *
                 sizeof(float));
      fv = new_fv;
    }
    input_frame->SetValue(FvSpec::GetUniqueID(spec), fv);
  }
  long time_elapsed =
      (boost::posix_time::microsec_clock::local_time() - start_time)
          .total_microseconds();
  input_frame->SetValue<long>("fv_gen.micros", time_elapsed);
  PushFrame(SINK_NAME, std::move(input_frame));
}
