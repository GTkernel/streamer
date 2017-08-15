
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
  EXPECT_EQ(PROCESSOR_TYPE_ENCODER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_ENCODER)));
#ifdef USE_CAFFE
  EXPECT_EQ(PROCESSOR_TYPE_FACENET,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FACENET)));
#endif  // USE_CAFFE
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
#ifdef USE_CAFFE
  EXPECT_EQ(PROCESSOR_TYPE_MTCNN_FACE_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_MTCNN_FACE_DETECTOR)));
#endif  // USE_CAFFE
#ifdef USE_NCS
  EXPECT_EQ(PROCESSOR_TYPE_NCS_YOLO_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_NCS_YOLO_DETECTOR)));
#endif  // USE_NCS
  EXPECT_EQ(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_NEURAL_NET_EVALUATOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OBJ_TRACKER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_OBJ_TRACKER)));
#ifdef USE_FRCNN
  EXPECT_EQ(PROCESSOR_TYPE_OBJECT_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_DETECTOR)));
#endif  // USE_FRCNN
  EXPECT_EQ(PROCESSOR_TYPE_OBJECT_TRACKER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_OBJECT_TRACKER)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_FACE_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_FACE_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_MOTION_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_PEOPLE_DETECTOR)));
#ifdef USE_SSD
  EXPECT_EQ(PROCESSOR_TYPE_SSD_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_SSD_DETECTOR)));
#endif  // USE_SSD
  EXPECT_EQ(PROCESSOR_TYPE_THROTTLER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_THROTTLER)));
#ifdef USE_CAFFE
  EXPECT_EQ(PROCESSOR_TYPE_YOLO_DETECTOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_YOLO_DETECTOR)));
#endif  // USE_CAFFE
  EXPECT_EQ(PROCESSOR_TYPE_INVALID,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_INVALID)));
}
