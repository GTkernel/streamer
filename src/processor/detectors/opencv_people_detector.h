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

#ifndef STREAMER_PROCESSOR_DETECTORS_OPENCV_PEOPLE_DETECTOR_H_
#define STREAMER_PROCESSOR_DETECTORS_OPENCV_PEOPLE_DETECTOR_H_

#ifdef USE_CUDA
#include <opencv2/cudaobjdetect/cudaobjdetect.hpp>
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif  // USE_CUDA

#include "processor/processor.h"

class OpenCVPeopleDetector : public Processor {
 public:
  OpenCVPeopleDetector();
  static std::shared_ptr<OpenCVPeopleDetector> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
#ifdef USE_CUDA
  cv::cuda::HOG hog_;
#else
  cv::HOGDescriptor hog_;
#endif  // USE_CUDA
};

#endif  // STREAMER_PROCESSOR_DETECTORS_OPENCV_PEOPLE_DETECTOR_H_
