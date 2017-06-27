//
// Created by Ran Xian (xranthoar@gmail.com) on 10/28/16.
//

#include <gtest/gtest.h>
#include "stream/stream.h"

TEST(STREAM_TEST, BASIC_TEST) {
  std::shared_ptr<Stream> stream(new Stream);
  auto reader = stream->Subscribe();

  auto input_frame = std::make_unique<Frame>();
  input_frame->SetValue("image", cv::Mat(10, 20, CV_8UC3));
  stream->PushFrame(std::move(input_frame));

  auto output_frame = reader->PopFrame();
  const auto& image = output_frame->GetValue<cv::Mat>("image");
  EXPECT_EQ(image.rows, 10);
  EXPECT_EQ(image.cols, 20);

  reader->UnSubscribe();
}

TEST(STREAM_TEST, SUBSCRIBE_TEST) {
  std::shared_ptr<Stream> stream(new Stream);
  auto reader1 = stream->Subscribe();
  auto reader2 = stream->Subscribe();

  stream->PushFrame(std::move(std::make_unique<Frame>()));
  stream->PushFrame(std::move(std::make_unique<Frame>()));

  // Both readers can pop twice
  reader1->PopFrame();
  reader1->PopFrame();

  reader2->PopFrame();
  reader2->PopFrame();

  reader1->UnSubscribe();
  reader2->UnSubscribe();
}
