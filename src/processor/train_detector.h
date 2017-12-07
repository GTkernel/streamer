
#ifndef STREAMER_PROCESSOR_TRAIN_DETECTOR_H_
#define STREAMER_PROCESSOR_TRAIN_DETECTOR_H_

#include <memory>

#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "processor/processor.h"

class TrainDetector : public Processor {
 public:
  TrainDetector();
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
  size_t num_divid;
  size_t display_scalar;
  bool hasTrain;
  cv::Mat RoI_mask;
  cv::Ptr<cv::BackgroundSubtractor> pMOG;
};

#endif  // STREAMER_PROCESSOR_TRAIN_DETECTOR_H_
