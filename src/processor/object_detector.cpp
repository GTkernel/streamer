/**
 * Multi-target detection using FRCNN
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "object_detector.h"
#include "common/context.h"
#include "model/model_manager.h"
#ifdef USE_FRCNN
#include "frcnn_detector.h"
#endif
#ifdef USE_SSD
#include "ssd_detector.h"
#endif
#include "caffe_mtcnn.h"
#include "caffe_yolo_detector.h"
#include "ncs_yolo_detector.h"
#include "opencv_face_detector.h"

ObjectDetector::ObjectDetector(const std::string& type,
                               const std::vector<ModelDesc>& model_descs,
                               Shape input_shape, float confidence_threshold,
                               float idle_duration,
                               const std::set<std::string>& targets)
    : Processor(PROCESSOR_TYPE_OBJECT_DETECTOR, {"input"}, {"output"}),
      type_(type),
      model_descs_(model_descs),
      input_shape_(input_shape),
      confidence_threshold_(confidence_threshold),
      idle_duration_(idle_duration),
      targets_(targets) {}

std::shared_ptr<ObjectDetector> ObjectDetector::Create(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool ObjectDetector::Init() {
  bool result = false;
  if (type_ == "opencv-face") {
    detector_.reset(new OpenCVFaceDetector(model_descs_.at(0)));
    result = detector_->Init();
  } else if (type_ == "mtcnn-face") {
    detector_.reset(new MtcnnFaceDetector(model_descs_));
    result = detector_->Init();
  } else if (type_ == "yolo") {
    detector_.reset(new YoloDetector(model_descs_.at(0)));
    result = detector_->Init();
#ifdef USE_FRCNN
  } else if (type_ == "frcnn") {
    detector_.reset(new FrcnnDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif
#ifdef USE_SSD
  } else if (type_ == "ssd") {
    detector_.reset(new SsdDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif
#ifdef USE_NCS
  } else if (type_ == "ncs-yolo") {
    detector_.reset(new NcsYoloDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif
  } else {
    LOG(FATAL) << "Detector type " << type_ << " not supported.";
  }

  return result;
}

bool ObjectDetector::OnStop() { return true; }

void ObjectDetector::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - last_detect_time_;
  if (diff.count() >= idle_duration_) {
    auto image = frame->GetValue<cv::Mat>("image");
    CHECK(image.channels() == input_shape_.channel &&
          image.size[1] == input_shape_.width &&
          image.size[0] == input_shape_.height);

    auto result = detector_->Detect(image);
    std::vector<ObjectInfo> filtered_res;
    for (const auto& m : result) {
      if (m.confidence > confidence_threshold_) {
        if (targets_.empty()) {
          filtered_res.push_back(m);
        } else {
          auto it = targets_.find(m.tag);
          if (it != targets_.end()) filtered_res.push_back(m);
        }
      }
    }

    auto original_img = frame->GetValue<cv::Mat>("original_image");
    CHECK(!original_img.empty());
    std::vector<string> tags;
    std::vector<Rect> bboxes;
    std::vector<float> confidences;
    std::vector<FaceLandmark> face_landmarks;
    bool face_landmarks_flag = false;
    float scale_factor[] = {(float)original_img.size[1] / (float)image.size[1],
                            (float)original_img.size[0] / (float)image.size[0]};
    for (const auto& m : filtered_res) {
      tags.push_back(m.tag);
      cv::Rect cr = m.bbox;
      int x = cr.x * scale_factor[0];
      int y = cr.y * scale_factor[1];
      int w = cr.width * scale_factor[0];
      int h = cr.height * scale_factor[1];
      if (x < 0) x = 0;
      if (y < 0) y = 0;
      if ((x + w) > original_img.cols) w = original_img.cols - x;
      if ((y + h) > original_img.rows) h = original_img.rows - y;
      bboxes.push_back(Rect(x, y, w, h));
      confidences.push_back(m.confidence);

      if (m.face_landmark_flag) {
        FaceLandmark fl = m.face_landmark;
        for (auto& m : fl.x) m *= scale_factor[0];
        for (auto& m : fl.y) m *= scale_factor[1];
        face_landmarks.push_back(fl);
        face_landmarks_flag = true;
      }
    }

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("tags", tags);
    frame->SetValue("bounding_boxes", bboxes);
    frame->SetValue("confidences", confidences);
    if (face_landmarks_flag) frame->SetValue("face_landmarks", face_landmarks);
    PushFrame("output", std::move(frame));
    LOG(INFO) << "Object detection took " << timer.ElapsedMSec() << " ms";
  } else {
    PushFrame("output", std::move(frame));
  }
}
