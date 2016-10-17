//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#include <gtest/gtest.h>
#include "tx1dnn.h"

TEST(CAMERA_MANAGER_TEST, TEST_BASIC) {
  CameraManager &manager = CameraManager::GetInstance();
  Context &context = Context::GetContext();

  EXPECT_EQ(context.GetString(H264_DECODER_GST_ELEMENT), "avdec_h264");
  EXPECT_EQ(context.GetString(H264_ENCODER_GST_ELEMENT), "vtenc_h264");
}