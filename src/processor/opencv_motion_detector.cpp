
#include "opencv_motion_detector.h"

OpenCVMotionDetector::OpenCVMotionDetector(float threshold, float max_duration) 
    : Processor({"input"}, {"output"}),
      first_frame_(true),
      previous_pixels_(0),
      threshold_(threshold),
      max_duration_(max_duration) {}

bool OpenCVMotionDetector::Init() {
  mog2_.reset(new cv::BackgroundSubtractorMOG2());
  return true;
}

bool OpenCVMotionDetector::OnStop() {
  mog2_.reset();
  return true;
}

void OpenCVMotionDetector::Process() {
  auto frame = GetFrame<ImageFrame>("input");
  cv::Mat image = frame->GetImage();

  cv::Mat fore;
  (*mog2_)(image, fore);

  cv::erode(fore, fore, cv::Mat()); cv::dilate(fore, fore, cv::Mat());
  cv::erode(fore, fore, cv::Mat()); cv::dilate(fore, fore, cv::Mat());
  cv::erode(fore, fore, cv::Mat()); cv::dilate(fore, fore, cv::Mat());

  cv::Mat frameDelta;
  bool need_send = false;
  if (first_frame_) {
    first_frame_ = false;
    need_send = true;
  } else {
    cv::absdiff(fore, previous_fore_, frameDelta);
    int diff_pixels = GetPixels(frameDelta);
    if (diff_pixels >= (previous_pixels_*threshold_)) {
      //printf("motion happen, diff_pixels[%d]\n", diff_pixels);
      need_send = true;
    }
  }
  previous_fore_ = fore;
  previous_pixels_ = GetPixels(fore);

  //imshow("fore", fore);
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now-last_send_time_;
  if (need_send || (diff.count() >= max_duration_)) {
    last_send_time_ = now;
    PushFrame("output",
        new ImageFrame(image, frame->GetOriginalImage()));
  }
}

ProcessorType OpenCVMotionDetector::GetType() {
  return PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR;
}

int OpenCVMotionDetector::GetPixels(cv::Mat& image) {
  int pixels = 0;
  int nr= image.rows;
  int nc= image.cols * image.channels();
  for (int j=0; j<nr; j++) {
    uchar* data= image.ptr<uchar>(j);
    for (int i=0; i<nc; i++) {
      if (data[i]) pixels++;
    }
  }
  return pixels;
}
