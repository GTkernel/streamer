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
 * Multi-target tracking using dlib.
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_TRACKERS_DLIB_TRACKER_H_
#define STREAMER_PROCESSOR_TRACKERS_DLIB_TRACKER_H_

#include <dlib/dlib/image_processing.h>
#include <dlib/dlib/opencv.h>

#include "processor/trackers/object_tracker.h"

class DlibTracker : public BaseTracker {
 public:
  DlibTracker(const std::string& uuid, const std::string& tag)
      : BaseTracker(uuid, tag) {
    impl_.reset(new dlib::correlation_tracker());
  }
  virtual ~DlibTracker() {}
  virtual void Initialise(const cv::Mat& gray_image, cv::Rect bb) {
    dlib::array2d<unsigned char> dlibImageGray;
    dlib::assign_image(dlibImageGray,
                       dlib::cv_image<unsigned char>(gray_image));
    dlib::rectangle initBB(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
    impl_->start_track(dlibImageGray, initBB);
  }
  virtual bool IsInitialised() { return true; }
  virtual void Track(const cv::Mat& gray_image) {
    dlib::array2d<unsigned char> dlibImageGray;
    dlib::assign_image(dlibImageGray,
                       dlib::cv_image<unsigned char>(gray_image));
    impl_->update(dlibImageGray);
  }
  virtual cv::Rect GetBB() {
    auto r = impl_->get_position();
    return cv::Rect(r.left(), r.top(), r.right() - r.left(),
                    r.bottom() - r.top());
  }
  virtual std::vector<double> GetBBFeature() {
    return std::vector<double>(128, 0.f);
  }

 private:
  std::unique_ptr<dlib::correlation_tracker> impl_;
};

#endif  // STREAMER_PROCESSOR_TRACKERS_DLIB_TRACKER_H_
