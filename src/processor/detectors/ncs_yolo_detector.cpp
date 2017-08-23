/**
 * Multi-target detection using fathom stick
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "processor/detectors/ncs_yolo_detector.h"

#include "common/context.h"
#include "model/model_manager.h"
#include "utils/yolo_utils.h"

bool NcsYoloDetector::Init() {
  std::string weights_file = model_desc_.GetModelParamsPath();
  LOG(INFO) << "weights_file: " << weights_file;

  detector_.reset(new NCSManager(weights_file.c_str(), 448));
  CHECK(detector_->Open()) << "Failed to open NCSManager";

  std::string labelmap_file = model_desc_.GetLabelFilePath();
  voc_names_ = ReadVocNames(labelmap_file);

  LOG(INFO) << "NcsYoloDetector initialized";
  return true;
}

std::vector<ObjectInfo> NcsYoloDetector::Detect(const cv::Mat& image) {
  std::vector<float> result_vec;
  detector_->LoadImageAndGetResult(result_vec, image);
  std::vector<std::tuple<int, cv::Rect, float>> detections;
  get_detections(detections, result_vec, image.size(), voc_names_.size() - 1);

  std::vector<ObjectInfo> result;
  for (const auto& m : detections) {
    ObjectInfo object_info;
    object_info.tag = voc_names_.at(std::get<0>(m) + 1);
    object_info.bbox = std::get<1>(m);
    object_info.confidence = std::get<2>(m);
    result.push_back(object_info);
  }

  return result;
}
