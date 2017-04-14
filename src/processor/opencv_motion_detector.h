#ifndef STREAMER_OPENCV_MOTION_DETECTOR_H
#define STREAMER_OPENCV_MOTION_DETECTOR_H

#include <chrono>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include "common/common.h"
#include "processor.h"

class OpenCVMotionDetector : public Processor {
  public:
    OpenCVMotionDetector(float threshold = 0.5);
    virtual ProcessorType GetType() override;

  protected:
    virtual bool Init() override;
    virtual bool OnStop() override;
    virtual void Process() override;

  private:
    int GetPixels(cv::Mat& image);

  private:
    std::unique_ptr<cv::BackgroundSubtractorMOG2> mog2_;
    bool first_frame_;
    cv::Mat previous_fore_;
    int previous_pixels_;
    std::chrono::time_point<std::chrono::system_clock> last_send_time_;
    float threshold_;
};

#endif // STREAMER_OPENCV_MOTION_DETECTOR_H
