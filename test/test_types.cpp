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

#include <gtest/gtest.h>
#include <json/src/json.hpp>

#include "common/types.h"

// Verifies that Rect::ToJson() produces a correctly-formatted JSON object The
// resulting JSON should look like this:
//   {
//     "Rect": {
//       "px": 1,
//       "py": 2,
//       "width": 3,
//       "height": 4
//     }
//   }
TEST(TestTypes, TestRectToJson) {
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;

  Rect r(a, b, c, d);
  nlohmann::json j = r.ToJson();
  nlohmann::json rect_j = j.at("Rect");

  EXPECT_EQ(rect_j.at("px").get<int>(), a);
  EXPECT_EQ(rect_j.at("py").get<int>(), b);
  EXPECT_EQ(rect_j.at("width").get<int>(), c);
  EXPECT_EQ(rect_j.at("height").get<int>(), d);
}

// Verifies that Rect::Rect(nlohmann::json) creates a properly-initialized Rect
// struct from a JSON object. See TestRectToJson for details on the format of
// the JSON object.
TEST(TestTypes, TestJsonToRect) {
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;

  nlohmann::json rect_j;
  rect_j["px"] = a;
  rect_j["py"] = b;
  rect_j["width"] = c;
  rect_j["height"] = d;
  nlohmann::json j;
  j["Rect"] = rect_j;
  Rect r(j);

  EXPECT_EQ(r.px, a);
  EXPECT_EQ(r.py, b);
  EXPECT_EQ(r.width, c);
  EXPECT_EQ(r.height, d);
}

TEST(TestTypes, TestProcessorTypesStringConversion) {
  EXPECT_EQ(PROCESSOR_TYPE_BINARY_FILE_WRITER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_BINARY_FILE_WRITER)));
  EXPECT_EQ(PROCESSOR_TYPE_BUFFER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_BUFFER)));
  EXPECT_EQ(PROCESSOR_TYPE_CAMERA,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_CAMERA)));
  EXPECT_EQ(PROCESSOR_TYPE_COMPRESSOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_COMPRESSOR)));
  EXPECT_EQ(PROCESSOR_TYPE_CUSTOM,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_CUSTOM)));
  EXPECT_EQ(PROCESSOR_TYPE_DB_WRITER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_DB_WRITER)));
  EXPECT_EQ(PROCESSOR_TYPE_DIFF_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_DIFF_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_DISPLAY,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_DISPLAY)));
  EXPECT_EQ(PROCESSOR_TYPE_ENCODER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_ENCODER)));
  EXPECT_EQ(PROCESSOR_TYPE_FACE_TRACKER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FACE_TRACKER)));
#ifdef USE_CAFFE
  EXPECT_EQ(PROCESSOR_TYPE_FACENET,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FACENET)));
#endif  // USE_CAFFE
#ifdef USE_TENSORFLOW
  EXPECT_EQ(PROCESSOR_TYPE_IMAGEMATCH,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_IMAGEMATCH)));
#endif  // USE_TENSORFLOW
  EXPECT_EQ(PROCESSOR_TYPE_FLOW_CONTROL_ENTRANCE,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_FLOW_CONTROL_ENTRANCE)));
  EXPECT_EQ(PROCESSOR_TYPE_FLOW_CONTROL_EXIT,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FLOW_CONTROL_EXIT)));
#ifdef USE_RPC
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_RECEIVER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_RECEIVER)));
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_SENDER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_SENDER)));
#endif  // USE_RPC
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_PUBLISHER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_PUBLISHER)));
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_SUBSCRIBER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_SUBSCRIBER)));
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_WRITER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_WRITER)));
  EXPECT_EQ(PROCESSOR_TYPE_FV_GEN,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FV_GEN)));
  EXPECT_EQ(PROCESSOR_TYPE_IMAGE_CLASSIFIER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_IMAGE_CLASSIFIER)));
  EXPECT_EQ(PROCESSOR_TYPE_IMAGE_SEGMENTER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_IMAGE_SEGMENTER)));
  EXPECT_EQ(PROCESSOR_TYPE_IMAGE_TRANSFORMER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_IMAGE_TRANSFORMER)));
  EXPECT_EQ(PROCESSOR_TYPE_JPEG_WRITER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_JPEG_WRITER)));
  EXPECT_EQ(PROCESSOR_TYPE_KEYFRAME_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_KEYFRAME_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_NEURAL_NET_EVALUATOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OBJECT_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_OBJECT_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OBJECT_TRACKER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_OBJECT_TRACKER)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_STRIDER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_STRIDER)));
  EXPECT_EQ(PROCESSOR_TYPE_TEMPORAL_REGION_SELECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_TEMPORAL_REGION_SELECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_THROTTLER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_THROTTLER)));
  EXPECT_EQ(PROCESSOR_TYPE_INVALID,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_INVALID)));
}
