/**
 * Multi-target detection using SSD
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_DETECTORS_SSD_DETECTOR_H_
#define STREAMER_PROCESSOR_DETECTORS_SSD_DETECTOR_H_

#include <set>

#include <caffe/caffe.hpp>

#include "common/common.h"
#include "model/model.h"
#include "processor/detectors/object_detector.h"

namespace ssd {
class Detector {
 public:
  Detector(const string& model_file, const string& weights_file,
           const string& mean_file, const string& mean_value);

  std::vector<std::vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  size_t num_channels_;
  cv::Mat mean_;
};
}  // namespace ssd

class SsdDetector : public BaseDetector {
 public:
  SsdDetector(const ModelDesc& model_desc) : model_desc_(model_desc) {}
  virtual ~SsdDetector() {}
  virtual bool Init();
  virtual std::vector<ObjectInfo> Detect(const cv::Mat& image);

 private:
  std::string GetLabelName(int label) const;

 private:
  ModelDesc model_desc_;
  std::unique_ptr<ssd::Detector> detector_;
  caffe::LabelMap label_map_;
};

#endif  // STREAMER_PROCESSOR_DETECTORS_SSD_DETECTOR_H_
