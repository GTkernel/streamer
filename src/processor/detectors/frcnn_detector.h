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
