//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#include <gtest/gtest.h>
#include "tx1dnn.h"

TEST(CONTEXT_TEST, TEST_BASIC) {
  Context &context = Context::GetContext();

  // Sanaity check
  context.GetString(H264_DECODER_GST_ELEMENT);
  context.GetString(H264_ENCODER_GST_ELEMENT);
}