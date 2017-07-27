
#include <gtest/gtest.h>

#include "common/types.h"

TEST(TestTypes, TestProcessorTypesStringConversion) {
  EXPECT_EQ(PROCESSOR_TYPE_CAMERA,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_CAMERA)));
  EXPECT_EQ(PROCESSOR_TYPE_CAMERA,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_CAMERA)));
  EXPECT_EQ(PROCESSOR_TYPE_COMPRESSOR,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_COMPRESSOR)));
  EXPECT_EQ(PROCESSOR_TYPE_CUSTOM,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_CUSTOM)));
  EXPECT_EQ(PROCESSOR_TYPE_ENCODER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_ENCODER)));
  EXPECT_EQ(PROCESSOR_TYPE_FILE_WRITER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FILE_WRITER)));
#ifdef USE_RPC
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_RECEIVER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_RECEIVER)));
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_SENDER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_SENDER)));
#endif  // USE_RPC
#ifdef USE_ZMQ
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_PUBLISHER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_PUBLISHER)));
  EXPECT_EQ(PROCESSOR_TYPE_FRAME_SUBSCRIBER,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_FRAME_SUBSCRIBER)));
#endif  // USE_ZMQ
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
  EXPECT_EQ(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_NEURAL_NET_EVALUATOR)));
  EXPECT_EQ(PROCESSOR_TYPE_OPENCV_FACE_DETECTOR,
            GetProcessorTypeByString(GetStringForProcessorType(
                PROCESSOR_TYPE_OPENCV_FACE_DETECTOR)));
  EXPECT_EQ(PROCESSOR_TYPE_INVALID,
            GetProcessorTypeByString(
                GetStringForProcessorType(PROCESSOR_TYPE_INVALID)));
}
