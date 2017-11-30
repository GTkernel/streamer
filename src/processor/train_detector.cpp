
#include "processor/train_detector.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "input";
constexpr auto ROIMASK_NAME = "PATH_TO_ROI_MASK";

TrainDetector::TrainDetector()
    : Processor(PROCESSOR_TYPE_THROTTLER, {SOURCE_NAME}, {SINK_NAME}),
      num_divid(10),
      display_scalar(1),
      hasTrain(false) {}

std::shared_ptr<TrainDetector> TrainDetector::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void TrainDetector::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr TrainDetector::GetSink() { return Processor::GetSink(SINK_NAME); }

bool TrainDetector::Init() {
  LOG(INFO) << "Train detector initialized";
  cv::Ptr<cv::BackgroundSubtractor> pMOG =
      cv::createBackgroundSubtractorMOG2(1000, 16, false);
  hasTrain = false;
  RoI_mask = cv::imread(ROIMASK_NAME);
  return true;
}

bool TrainDetector::OnStop() { return true; }

void TrainDetector::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  const cv::Mat& img_input =
      frame->GetValue<cv::Mat>("image");  // original_image
  cv::Mat img_mask;
  // Foreground detection
  pMOG->cv::BackgroundSubtractor::apply(img_input, img_mask);

  // Apply RoI mask and count foreground pixels
  int width = img_input.cols / num_divid;
  float ratio = 0;
  for (size_t i = 0; i < num_divid; i++) {
    cv::Mat stripe_detect, white_mask, white_detect;
    cv::Mat stripe_mask(img_input.rows, img_input.cols, CV_8UC1,
                        cv::Scalar(0, 0, 0));
    cv::Mat stripe_roi =
        RoI_mask(cv::Rect(width * i, 0, width, img_input.rows));
    stripe_roi.cv::Mat::copyTo(
        stripe_mask(cv::Rect(width * i, 0, width, img_input.rows)));
    cv::findNonZero(stripe_mask, white_mask);
    img_mask.cv::Mat::copyTo(stripe_detect, stripe_mask);
    cv::findNonZero(stripe_detect, white_detect);
    ratio = cv::max(ratio, (float)white_detect.rows / white_mask.rows);
  }

  if (ratio > 0.5) {
    hasTrain = true;
  }
}
