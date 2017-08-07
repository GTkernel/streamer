/**
 * Multi-target detection using SSD
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_SSD_DETECTOR_H
#define STREAMER_SSD_DETECTOR_H

#include <caffe/caffe.hpp>
#include <set>
#include "common/common.h"
#include "model/model.h"
#include "processor.h"

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

class SsdDetector : public Processor {
 public:
  SsdDetector(const ModelDesc& model_desc, Shape input_shape,
              float confidence_threshold, float idle_duration = 0.f,
              const std::set<std::string>& targets = std::set<std::string>());
  static std::shared_ptr<SsdDetector> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string GetLabelName(int label) const;

 private:
  ModelDesc model_desc_;
  Shape input_shape_;
  std::unique_ptr<ssd::Detector> detector_;
  float confidence_threshold_;
  float idle_duration_;
  std::chrono::time_point<std::chrono::system_clock> last_detect_time_;
  std::set<std::string> targets_;
  caffe::LabelMap label_map_;
};

#endif  // STREAMER_SSD_DETECTOR_H
