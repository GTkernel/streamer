/**
 * Multi-target detection using FRCNN
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_DETECTORS_FRCNN_DETECTOR_H_
#define STREAMER_PROCESSOR_DETECTORS_FRCNN_DETECTOR_H_

#include <api/api.hpp>

#include "model/model.h"
#include "processor/processor.h"

class FrcnnDetector : public BaseDetector {
 public:
  FrcnnDetector(const ModelDesc& model_desc) : model_desc_(model_desc) {}
  virtual ~FrcnnDetector() {}
  virtual bool Init();
  virtual std::vector<ObjectInfo> Detect(const cv::Mat& image);

 private:
  ModelDesc model_desc_;
  std::unique_ptr<API::Detector> detector_;
};

#endif  // STREAMER_PROCESSOR_DETECTORS_FRCNN_DETECTOR_H_
