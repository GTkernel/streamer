
#ifndef STREAMER_PROCESSOR_TRAIN_DETECTOR_H_
#define STREAMER_PROCESSOR_TRAIN_DETECTOR_H_

//#include "cv.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <memory>
#include "common/types.h"
#include "processor/processor.h"



// A processor that a stream's framerate to a specified maximum.
class TrainDetector : public Processor {
 public:
  TrainDetector();
  static std::shared_ptr<TrainDetector> Create(const FactoryParamsType& params);

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
