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
/**
 * Multi-target tracking processors
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "processor/trackers/object_tracker.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "common/context.h"
#ifdef USE_DLIB
#include "processor/trackers/dlib_tracker.h"
#endif  // USE_DLIB

ObjectTracker::ObjectTracker(const std::string& type,
                             float calibration_duration)
    : Processor(PROCESSOR_TYPE_OBJECT_TRACKER, {"input"}, {"output"}),
      type_(type),
      calibration_duration_(calibration_duration) {}

std::shared_ptr<ObjectTracker> ObjectTracker::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool ObjectTracker::Init() {
  LOG(INFO) << "ObjectTracker initialized";
  return true;
}

bool ObjectTracker::OnStop() {
  tracker_list_.clear();
  return true;
}

void ObjectTracker::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");
  auto image = frame->GetValue<cv::Mat>("original_image");
  if (image.channels() == 3) {
    cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
  } else {
    gray_image_ = image;
  }

  std::vector<Rect> tracked_bboxes;
  std::vector<std::string> tracked_tags;
  std::vector<std::string> tracked_uuids;
  std::vector<std::vector<double>> features;
  if (frame->Count("bounding_boxes") > 0) {
    auto bboxes = frame->GetValue<std::vector<Rect>>("bounding_boxes");
    LOG(INFO) << "Got new MetadataFrame, bboxes size is " << bboxes.size()
              << ", current tracker size is " << tracker_list_.size();
    std::vector<Rect> untracked_bboxes = bboxes;
    auto untracked_tags = frame->GetValue<std::vector<std::string>>("tags");
    CHECK(untracked_bboxes.size() == untracked_tags.size());
    for (auto it = tracker_list_.begin(); it != tracker_list_.end();) {
      (*it)->Track(gray_image_);
      cv::Rect rt((*it)->GetBB());
      float best_percent = 0.f;
      for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
        cv::Rect ru(untracked_bboxes[i].px, untracked_bboxes[i].py,
                    untracked_bboxes[i].width, untracked_bboxes[i].height);
        cv::Rect intersects = rt & ru;
        float percent = (float)intersects.area() / (float)ru.area();
        if (percent >= 0.7) {
          untracked_bboxes.erase(untracked_bboxes.begin() + i);
          untracked_tags.erase(untracked_tags.begin() + i);
          best_percent = percent;
          break;
        }
      }
      if (best_percent >= 0.7) {
        tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
        features.push_back((*it)->GetBBFeature());
        it++;
      } else {
        LOG(INFO) << "Remove tracker, best_percent is " << best_percent;
        tracker_list_.erase(it++);
      }
    }

    CHECK(untracked_bboxes.size() == untracked_tags.size());
    for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
      LOG(INFO) << "Create new tracker";
      int x = untracked_bboxes[i].px;
      int y = untracked_bboxes[i].py;
      int w = untracked_bboxes[i].width;
      int h = untracked_bboxes[i].height;
      CHECK((x >= 0) && (y >= 0) && (x + w <= gray_image_.cols) &&
            (y + h <= gray_image_.rows));
      cv::Rect bb(x, y, w, h);
      boost::uuids::uuid uuid = boost::uuids::random_generator()();
      std::string uuid_str = boost::lexical_cast<std::string>(uuid);
      std::shared_ptr<BaseTracker> new_tracker;
      if (type_ == "dlib") {
#ifdef USE_DLIB
        new_tracker.reset(new DlibTracker(uuid_str, untracked_tags[i]));
#else
        LOG(FATAL) << "Tracker type " << type_
                   << " not supported, please compile with -DUSE_DLIB=ON";
#endif  // USE_DLIB
      } else {
        LOG(FATAL) << "Tracker type " << type_ << " not supported.";
      }
      new_tracker->Initialise(gray_image_, bb);
      CHECK(new_tracker->IsInitialised());
      new_tracker->Track(gray_image_);
      cv::Rect rt(new_tracker->GetBB());
      tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
      tracked_tags.push_back(untracked_tags[i]);
      tracked_uuids.push_back(uuid_str);
      features.push_back(new_tracker->GetBBFeature());
      tracker_list_.push_back(new_tracker);
    }
    last_calibration_time_ = std::chrono::system_clock::now();
  } else {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = now - last_calibration_time_;
    if (diff.count() < calibration_duration_) {
      for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ++it) {
        (*it)->Track(gray_image_);
        cv::Rect rt((*it)->GetBB());
        tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
        features.push_back((*it)->GetBBFeature());
      }
    } else {
      LOG(INFO) << "Time " << calibration_duration_
                << " is up, need calibration ......";
      return;
    }
  }

  frame->SetValue("bounding_boxes", tracked_bboxes);
  frame->SetValue("tags", tracked_tags);
  frame->SetValue("uuids", tracked_uuids);
  frame->SetValue("features", features);
  PushFrame("output", std::move(frame));
  LOG(INFO) << "ObjectTracker took " << timer.ElapsedMSec() << " ms";
}
