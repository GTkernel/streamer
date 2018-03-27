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

#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "processor/strider.h"
#include "stream/frame.h"
#include "stream/stream.h"

TEST(TestStrider, TestBasic) {
  unsigned long num_output_frames = 5;
  unsigned long stride = 10;

  auto strider = std::make_shared<Strider>(stride);
  auto stream = std::make_shared<Stream>();
  strider->SetSource(stream);

  unsigned long num_total_frames = num_output_frames * stride;
  auto reader = strider->GetSink()->Subscribe(num_total_frames);
  strider->Start(num_total_frames);

  for (unsigned long i = 0; i < num_total_frames; ++i) {
    auto frame = std::make_unique<Frame>();
    frame->SetValue("frame_id", i);
    stream->PushFrame(std::move(frame));
  }

  for (unsigned long i = 0; i < num_output_frames; ++i) {
    unsigned long expected_id = i * stride;
    LOG(INFO) << "Waiting for frame: " << expected_id;
    auto id = reader->PopFrame()->GetValue<unsigned long>("frame_id");
    ASSERT_EQ(expected_id, id);
  }

  reader->UnSubscribe();
  strider->Stop();
}
