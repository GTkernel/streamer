
#include "processor/train_detector.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

TrainDetector::TrainDetector(
    unsigned long num_leading_frames, unsigned long num_trailing_frames,
    const std::string& roi_mask_path, double threshold, unsigned int num_div,
    double width_init, double width_scalar, unsigned int roi_mask_offset_X,
    unsigned int roi_mask_offset_Y, unsigned int roi_mask_offset_width,
    unsigned int roi_mask_offset_height)
    : Processor(PROCESSOR_TYPE_TRAIN_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      buffer_{num_leading_frames},
      num_trailing_frames_(num_trailing_frames),
      roi_mask_(cv::imread(roi_mask_path)),
      num_div_(num_div),
      width_init_(width_init),
      width_scalar_(width_scalar),
      roi_mask_offset_X_(roi_mask_offset_X),
      roi_mask_offset_Y_(roi_mask_offset_Y),
      roi_mask_offset_width_(roi_mask_offset_width),
      roi_mask_offset_height_(roi_mask_offset_height),
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
  bool is_FalsePositive = isFalsePositive(image);
  // TODO: need to intergrate is_FalsePositive flag of multiple frames to decide
  // whether to keep buffer or not

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
  // Prepare ROI masks for stripes of sub-region
  // This code segment may be moved to intialzation, rather than here and run
  // for every frame
  roi_mask_cropped_ =
      roi_mask_(cv::Rect(roi_mask_offset_X_, roi_mask_offset_Y_,
                         roi_mask_offset_width_, roi_mask_offset_height_));
  int widths[num_div_];  // width of each stripe
  int width_start = 0;
  int num_pixel_mask[num_div_];  // num of foreground pixels in mask
  std::vector<cv::Mat> stripe_masks;
  for (decltype(num_div_) i = 0; i < num_div_; ++i) {
    if (i > 0) {
      width_start += widths[i - 1];
    }
    widths[i] = (int)round(image.cols * width_init_ * pow(width_scalar_, i));
    if (widths[i] > image.cols - width_start) break;
    cv::Mat white_mask;
    cv::Mat stripe_mask(image.rows, widths[i], CV_8UC1, cv::Scalar(0, 0, 0));

    stripe_mask =
        roi_mask_cropped_(cv::Rect(width_start, 0, widths[i], image.rows));
    stripe_masks.push_back(stripe_mask);
    cv::findNonZero(stripe_mask, white_mask);
    num_pixel_mask[i] = white_mask.rows;
  }

  cv::Mat img_mask;
  // Foreground detection
  pmog_->apply(image, img_mask);

  // Apply RoI masks on foreground detection
  float ratio = 0;
  width_start = 0;
  for (decltype(num_div_) i = 0; i < num_div_; ++i) {
    if (i > 0) {
      width_start += widths[i - 1];
    }
    if (widths[i] > image.cols - width_start) {
      break;
    }
    cv::Mat white_detect;
    cv::Mat stripe_detect(image.rows, widths[i], CV_8UC1, cv::Scalar(0, 0, 0));
    stripe_detect = img_mask(cv::Rect(width_start, 0, widths[i], image.rows));
    cv::Mat stripe_detect_masked;
    stripe_detect.Mat::copyTo(stripe_detect_masked, stripe_masks[i]);
    cv::findNonZero(stripe_detect_masked, white_detect);
    // Get ratio of detected foreground pixels to that of the mask
    ratio = cv::max(ratio, (float)white_detect.rows / num_pixel_mask[i]);
  }
  return ratio > threshold_;
}

bool TrainDetector::isFalsePositive(const cv::Mat& image) {
  cv::Mat flow;
  cv::UMat flowUmat, prevgray;  // UMat uses GPU to bring potential speed-up
  cv::Mat gray;
  image.Mat::copyTo(gray);
  cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

  // Skip the first frame
  if (prevgray.empty()) {
    return false;
  }
  // Calculate optical flow
  cv::calcOpticalFlowFarneback(prevgray, gray, flowUmat, 0.3, 3, 20, 2, 5, 1.1,
                               cv::OPTFLOW_USE_INITIAL_FLOW);
  flowUmat.UMat::copyTo(flow);
  std::vector<cv::Mat1f> OF;
  cv::split(flow, OF);
  cv::Mat magnitude, angle;
  cv::cartToPolar(OF[0], OF[1], magnitude, angle, true);

  int hbins = 18;  // number of bins in histogram
  int idx_channel = 0;
  float range[] = {0, 360};  // the upper boundary is exclusive
  const float* histRange = {range};
  cv::Mat angle_hist;
  cv::calcHist(&angle, 1, &idx_channel, roi_mask_cropped_, angle_hist, 1,
               &hbins, &histRange,
               true,  // the histogram is uniform
               false);
  gray.Mat::copyTo(prevgray);

  // TODO: compare distance between HOF from adjacent frames
  // Return false if distance < threshold, true otherwise

  return false;
}