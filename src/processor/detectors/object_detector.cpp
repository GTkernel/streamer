// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Multi-target detection using FRCNN

#include "processor/detectors/object_detector.h"

#include "common/context.h"
#include "model/model_manager.h"
#ifdef USE_CAFFE
#include "processor/detectors/caffe_mtcnn_face_detector.h"
#include "processor/detectors/caffe_yolo_detector.h"
#endif  // USE_CAFFE
#ifdef USE_FRCNN
#include "processor/detectors/frcnn_detector.h"
#endif  // USE_FRCNN
#ifdef USE_NCS
#include "processor/detectors/ncs_yolo_detector.h"
#endif  // USE_NCS
#include "processor/detectors/opencv_face_detector.h"
#ifdef USE_SSD
#include "processor/detectors/ssd_detector.h"
#endif  // USE_SSD

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
#ifdef USE_CAFFE
  } else if (type_ == "mtcnn-face") {
    detector_.reset(new MtcnnFaceDetector(model_descs_));
    result = detector_->Init();
  } else if (type_ == "yolo") {
    detector_.reset(new YoloDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif  // USE_CAFFE
#ifdef USE_FRCNN
  } else if (type_ == "frcnn") {
    detector_.reset(new FrcnnDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif  // USE_FRCNN
#ifdef USE_SSD
  } else if (type_ == "ssd") {
    detector_.reset(new SsdDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif  // USE_SSD
#ifdef USE_NCS
  } else if (type_ == "ncs-yolo") {
    detector_.reset(new NcsYoloDetector(model_descs_.at(0)));
    result = detector_->Init();
#endif  // USE_NCS
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
    std::vector<std::string> tags;
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
