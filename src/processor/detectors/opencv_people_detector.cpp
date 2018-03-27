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

#include "processor/detectors/opencv_people_detector.h"

OpenCVPeopleDetector::OpenCVPeopleDetector()
    : Processor(PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR, {"input"}, {"output"}) {}

std::shared_ptr<OpenCVPeopleDetector> OpenCVPeopleDetector::Create(
    const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool OpenCVPeopleDetector::Init() {
  hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  return true;
}

bool OpenCVPeopleDetector::OnStop() { return true; }

void OpenCVPeopleDetector::Process() {
  auto frame = GetFrame("input");
  const cv::Mat& image = frame->GetValue<cv::Mat>("image");

  std::vector<cv::Rect> results;

  std::vector<Rect> results_rect;

#ifdef USE_CUDA
  cv::gpu::GpuMat image_gpu(image);

  hog_.detectMultiScale(image_gpu, results, 0, cv::Size(8, 8), cv::Size(32, 32),
                        1.05, 2);
#else
  hog_.detectMultiScale(image, results, 0, cv::Size(8, 8), cv::Size(32, 32),
                        1.05, 2);
#endif  // USE_CUDA
  for (auto result : results) {
    results_rect.emplace_back(result.x, result.y, result.width, result.height);
  }

  frame->SetValue("bounding_boxes", results_rect);
  PushFrame("output", std::move(frame));
}
