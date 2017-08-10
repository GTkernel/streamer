/**
 * Multi-target detection using fathom stick
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "ncs_yolo_detector.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "utils/yolo_utils.h"

NcsYoloDetector::NcsYoloDetector(const ModelDesc& model_desc, Shape input_shape,
                                 float confidence_threshold,
                                 float idle_duration,
                                 const std::set<std::string>& targets)
    : Processor(PROCESSOR_TYPE_NCS_YOLO_DETECTOR, {"input"}, {"output"}),
      model_desc_(model_desc),
      input_shape_(input_shape),
      confidence_threshold_(confidence_threshold),
      idle_duration_(idle_duration),
      targets_(targets) {}

std::shared_ptr<NcsYoloDetector> NcsYoloDetector::Create(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

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

bool NcsYoloDetector::OnStop() { return true; }

void NcsYoloDetector::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");

  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - last_detect_time_;
  if (diff.count() >= idle_duration_) {
    auto img = frame->GetValue<cv::Mat>("image");
    CHECK(img.channels() == input_shape_.channel &&
          img.size[1] == input_shape_.width &&
          img.size[0] == input_shape_.height);
    detector_->LoadImage(img);
    std::vector<float> result;
    detector_->GetResult(result);
    std::vector<std::tuple<int, cv::Rect, float>> detections;
    get_detections(detections, result, img.size(), label_map_.item_size() - 1);
    std::vector<std::tuple<int, cv::Rect, float>> filtered_res;
    for (int i = 0; i < detections.size(); ++i) {
      const auto& d = detections[i];
      const float score = std::get<2>(d);
      if (score >= confidence_threshold_) {
        if (targets_.empty()) {
          filtered_res.push_back(d);
        } else {
          auto it = targets_.find(voc_names_.at(std::get<0>(d) + 1));
          if (it != targets_.end()) filtered_res.push_back(d);
        }
      }
    }

    auto original_img = frame->GetValue<cv::Mat>("original_image");
    CHECK(!original_img.empty());
    std::vector<string> tags;
    std::vector<Rect> bboxes;
    std::vector<float> confidences;
    float scale_factor[] = {(float)original_img.size[1] / (float)img.size[1],
                            (float)original_img.size[0] / (float)img.size[0]};
    for (const auto& m : filtered_res) {
      const cv::Rect& r = std::get<1>(m);
      int xmin = r.x * scale_factor[0];
      int ymin = r.y * scale_factor[1];
      int xmax = (r.x + r.width) * scale_factor[0];
      int ymax = (r.y + r.height) * scale_factor[1];
      if (xmin < 0) xmin = 0;
      if (ymin < 0) ymin = 0;
      if (xmax > original_img.cols) xmax = original_img.cols;
      if (ymax > original_img.rows) ymax = original_img.rows;

      tags.push_back(voc_names_.at(std::get<0>(m) + 1));
      bboxes.push_back(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
      confidences.push_back(std::get<2>(m));
    }

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("tags", tags);
    frame->SetValue("bounding_boxes", bboxes);
    frame->SetValue("confidences", confidences);
    PushFrame("output", std::move(frame));
    LOG(INFO) << "NcsYolo detection took " << timer.ElapsedMSec() << " ms";
  } else {
    PushFrame("output", std::move(frame));
  }
}
