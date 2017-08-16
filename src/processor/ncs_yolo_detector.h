/**
 * Multi-target detection using fathom stick
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_
#define STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_

#include <caffe/caffe.hpp>
#include <set>
#include "common/common.h"
#include "model/model.h"
#include "ncs/ncs.h"
#include "object_detector.h"
#include "processor.h"

class NcsYoloDetector : public BaseDetector {
 public:
  NcsYoloDetector(const ModelDesc& model_desc) : model_desc_(model_desc) {}
  virtual ~NcsYoloDetector() {}
  virtual bool Init();
  virtual std::vector<ObjectInfo> Detect(const cv::Mat& image);

 private:
  ModelDesc model_desc_;
  std::unique_ptr<NCSManager> detector_;
  std::vector<std::string> voc_names_;
};

#endif  // STREAMER_PROCESSOR_NCS_YOLO_DETECTOR_H_
