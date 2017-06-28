//
// Created by Ran Xian (xranthoar@gmail.com) on 10/28/16.
//

#include <gtest/gtest.h>
#include "stream/stream.h"

TEST(STREAM_TEST, BASIC_TEST) {
  std::shared_ptr<Stream> stream(new Stream);
  auto reader = stream->Subscribe();
  stream->PushFrame(new BytesFrame(std::vector<char>()));
  EXPECT_EQ(reader->PopFrame<BytesFrame>()->GetType(), FRAME_TYPE_BYTES);

  stream->PushFrame(new BytesFrame(std::vector<char>(), cv::Mat()));
  EXPECT_EQ(reader->PopFrame()->GetType(), FRAME_TYPE_BYTES);

  stream->PushFrame(new ImageFrame(cv::Mat(), cv::Mat(10, 20, CV_8UC3)));
  auto image_frame = reader->PopFrame<ImageFrame>();
  EXPECT_EQ(image_frame->GetOriginalImage().rows, 10);
  EXPECT_EQ(image_frame->GetOriginalImage().cols, 20);

  reader->UnSubscribe();
}

TEST(STREAM_TEST, SUBSCRIBE_TEST) {
  std::shared_ptr<Stream> stream(new Stream);
  auto reader1 = stream->Subscribe();
  auto reader2 = stream->Subscribe();

  stream->PushFrame(new BytesFrame(std::vector<char>()));
  stream->PushFrame(new ImageFrame(cv::Mat()));

  // Both readers can pop twice
  reader1->PopFrame();
  reader1->PopFrame();

  reader2->PopFrame();
  reader2->PopFrame();

  reader1->UnSubscribe();
  reader2->UnSubscribe();
}
