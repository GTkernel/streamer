
#include "processor/train_detector.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

TrainDetector::TrainDetector(unsigned long num_leading_frames,
                             unsigned long num_trailing_frames,
                             const std::string& roi_mask_path, double threshold,
                             unsigned int num_div)
    : Processor(PROCESSOR_TYPE_TRAIN_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      buffer_{num_leading_frames},
      num_trailing_frames_(num_trailing_frames),
      roi_mask_(cv::imread(roi_mask_path)),
      num_div_(num_div),
      pmog_(cv::createBackgroundSubtractorMOG2(1000, 16, false)),
      threshold_(threshold),
      previous_has_train_(false) {}

std::shared_ptr<TrainDetector> TrainDetector::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void TrainDetector::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr TrainDetector::GetSink() { return Processor::GetSink(SINK_NAME); }

bool TrainDetector::Init() { return true; }

bool TrainDetector::OnStop() { return true; }

void TrainDetector::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  const cv::Mat& image = frame->GetValue<cv::Mat>("image");
  // Run train detection algorithm.
  bool has_train = HasTrain(image);

  if (has_train) {
    // We detected a train, so we should push the current frame and any frames
    // in the buffer.
    PushFrame(SINK_NAME, std::move(frame));
    for (auto& frame : buffer_) {
      PushFrame(SINK_NAME, std::move(frame));
    }
    buffer_.clear();
  } else if (previous_has_train_) {
    // A train just disappeared, so we need to send "num_trailing_frames_" more
    // frames.
    if (num_trailing_frames_ > 0) {
      PushFrame(SINK_NAME, std::move(frame));
      num_remaining_frames_ = num_trailing_frames_ - 1;
    }
  } else if (num_remaining_frames_ > 0) {
    // A train disappear recently, so we still need to send this frame.
    PushFrame(SINK_NAME, std::move(frame));
    --num_remaining_frames_;
  } else {
    // Add the frame to the buffer. If the buffer is full, the oldest frame is
    // automatically discarded.
    buffer_.push_back(std::move(frame));
  }

  previous_has_train_ = has_train;
}

bool TrainDetector::HasTrain(const cv::Mat& image) {
  cv::Mat img_mask;
  // Foreground detection
  pmog_->apply(image, img_mask);

  // Apply RoI mask and count foreground pixels
  int width = image.cols / num_div_;
  float ratio = 0;
  for (decltype(num_div_) i = 0; i < num_div_; ++i) {
    cv::Mat stripe_detect, white_mask, white_detect;
    cv::Mat stripe_mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::Mat stripe_roi = roi_mask_(cv::Rect(width * i, 0, width, image.rows));
    stripe_roi.cv::Mat::copyTo(
        stripe_mask(cv::Rect(width * i, 0, width, image.rows)));
    cv::findNonZero(stripe_mask, white_mask);
    img_mask.cv::Mat::copyTo(stripe_detect, stripe_mask);
    cv::findNonZero(stripe_detect, white_detect);
    ratio = cv::max(ratio, (float)white_detect.rows / white_mask.rows);
  }

  return ratio > threshold_;
}
