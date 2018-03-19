
#include "processor/fv_gen.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FvSpec::FvSpec(const std::string& layer_name, int xmin, int ymin, int xmax,
               int ymax, bool flat)
    : layer_name_(layer_name),
      roi_(xmin, ymin, xmax - xmin, ymax - ymin),
      yrange_(ymin, ymax),
      xrange_(xmin, xmax),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      flat_(flat) {
  if (ymin == 0 && ymax == 0) {
    LOG(INFO) << "No bounds specified for Feature Vector vertical axis, "
                 "using full output";
    yrange_ = cv::Range::all();
  }
  if (xmin == 0 && xmax == 0) {
    LOG(INFO) << "No bounds specified for Feature Vector horizontal axis, "
                 "using full output";
    xrange_ = cv::Range::all();
  }
}

std::string FvSpec::GetUniqueID(const FvSpec& spec) {
  std::ostringstream ss;
  ss << spec.layer_name_ << spec.xmin_ << spec.ymin_ << spec.xmax_ << spec.ymax_
     << std::boolalpha << spec.flat_;
  return ss.str();
}

FvGen::FvGen() : Processor(PROCESSOR_TYPE_FV_GEN, {SOURCE_NAME}, {SINK_NAME}) {}

std::shared_ptr<FvGen> FvGen::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void FvGen::AddFv(const std::string& layer_name, int xmin, int xmax, int ymin,
                  int ymax, bool flat) {
  feature_vector_specs_.push_back(
      FvSpec(layer_name, xmin, xmax, ymin, ymax, flat));
}

void FvGen::SetSource(StreamPtr stream) { SetSource(SOURCE_NAME, stream); }

StreamPtr FvGen::GetSink() { return Processor::GetSink(SINK_NAME); }

bool FvGen::Init() { return true; }

bool FvGen::OnStop() { return true; }

void FvGen::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  for (const auto& spec : feature_vector_specs_) {
    cv::Mat input_mat = frame->GetValue<cv::Mat>(spec.layer_name_);
    cv::Mat fv;
    if (spec.roi_.height != 0 && spec.roi_.width != 0) {
      fv = input_mat({spec.yrange_, spec.xrange_});

#ifdef MODE_VERIFY
      // Keep this code because it is very useful for debugging.
      std::vector<cv::Mat> channels;
      int full_height = input_mat.size[0];
      int full_width = input_mat.size[1];
      int roi_height = spec.roi_.height;
      int roi_width = spec.roi_.width;
      int channels = input_mat.channels();
      LOG(INFO) << full_height << " " << full_width << " " << channels;
      for (decltype(channels) c = 0; c < channels; ++c) {
        for (decltype(spec.ymax_) y = spec.ymin_; y < spec.ymax_; ++y) {
          for (decltype(spec.xmax_) x = spec.xmin_; x < spec.xmax_; ++x) {
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
#endif  // MODE_VERIFY
    } else {
      fv = input_mat;
    }
    if (spec.flat_) {
      // Suspected heap corruption occurring somewhere causing possible segfault
      // in this block
      cv::Mat new_fv =
          cv::Mat(spec.roi_.height * spec.roi_.width * input_mat.channels(), 1,
                  CV_32FC1);
      memcpy(new_fv.data, fv.clone().data,
             spec.roi_.height * spec.roi_.width * input_mat.channels() * 0 *
                 sizeof(float));
      fv = new_fv;
    }
    frame->SetValue(FvSpec::GetUniqueID(spec), fv);
  }

  PushFrame(SINK_NAME, std::move(frame));
}
