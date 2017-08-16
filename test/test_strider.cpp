#include <gtest/gtest.h>
#include "stream/stream.h"
#include "processor/strider.h"

TEST(StriderTest, TestBasic) {
  unsigned long stride = 10;
  auto stream = std::make_shared<Stream>();

  Strider strider(stride);
  strider.SetSource(stream);
  auto reader = strider.GetSink("output")->Subscribe();
  strider.Start();

  for (unsigned long i = 0; i < 50; i++) {
    auto frame = std::make_unique<Frame>();
    frame->SetValue("frame_id", i);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stream->PushFrame(std::move(frame));
  }

  for (unsigned long i = 0; i < 5; i++) {
    auto current = i * stride;
    LOG(INFO) << "Waiting for frame_id = " << current;
    auto id = reader->PopFrame()->GetValue<unsigned long>("frame_id");
    ASSERT_EQ(id, current);
  }

  reader->UnSubscribe();
  strider.Stop();
}
