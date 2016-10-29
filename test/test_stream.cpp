//
// Created by Ran Xian (xranthoar@gmail.com) on 10/28/16.
//

#include <gtest/gtest.h>
#include "stream/stream.h"

TEST(STREAM_TEST, BASIC_TEST) {
  std::shared_ptr<Stream> stream(new Stream);
  stream->PushFrame(new BytesFrame(DataBuffer()));
  EXPECT_EQ(stream->PopFrame<BytesFrame>()->GetType(), FRAME_TYPE_BYTES);

  stream->PushFrame(new BytesFrame(DataBuffer(), cv::Mat()));
  EXPECT_EQ(stream->PopFrame()->GetType(), FRAME_TYPE_BYTES);

  stream->PushFrame(new ImageFrame(cv::Mat(), cv::Mat(10, 20, CV_8UC3)));
  auto image_frame = stream->PopFrame<ImageFrame>();
  EXPECT_EQ(image_frame->GetOriginalImage().rows, 10);
  EXPECT_EQ(image_frame->GetOriginalImage().cols, 20);
}