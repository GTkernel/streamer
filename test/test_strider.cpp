
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "processor/strider.h"
#include "stream/frame.h"
#include "stream/stream.h"

TEST(StriderTest, TestBasic) {
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
