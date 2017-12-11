
#ifndef STREAMER_PROCESSOR_TRAIN_DETECTOR_H_
#define STREAMER_PROCESSOR_TRAIN_DETECTOR_H_

#include <memory>

#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "processor/processor.h"

class TrainDetector : public Processor {
 public:
  TrainDetector(unsigned long num_leading_frames,
                unsigned long num_trailing_frames,
                const std::string& roi_mask_path, double threshold = 0.5,
                unsigned int num_div = 10);
  static std::shared_ptr<TrainDetector> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // Returns whether the provided image contains a train.
  bool HasTrain(const cv::Mat& image);

  // Stores recent frames, which will be pushed only if a train is detected.
  // This is important because when a train is detected, we also want a few
  // frames from before the train appeared.
  boost::circular_buffer<std::unique_ptr<Frame>> buffer_;
  // The number of frames to push after a train disappears. This is important
  // because after a train passes, we want to push a few more frames for
  // additional air quality analysis.
  unsigned long num_trailing_frames_;
  // A counter that is used to track the number of frames that still need to be
  // sent after that last train disappeared.
  unsigned long num_remaining_frames_;
  cv::Mat roi_mask_;
  unsigned int num_div_;
  cv::Ptr<cv::BackgroundSubtractor> pmog_;
  // The ratio of pixel change above which the frame is considered to contain a
  // train.
  double threshold_;
  bool previous_has_train_;
};

#endif  // STREAMER_PROCESSOR_TRAIN_DETECTOR_H_
