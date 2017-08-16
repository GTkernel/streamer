#include <gtest/gtest.h>
#include "camera/gst_camera.h"
#include "processor/strider.h"

TEST(StriderTest, TestBasic) {
  auto stride = 10;
  auto camera_name = "TEST_CAMERA";
  auto video_uri = "gst://videotestsrc ! video/x-raw,width=640,height=480";
  GSTCamera camera(camera_name, video_uri, 640, 480);

  Strider strider(stride);
  strider.SetSource(camera.GetStream());

  auto reader = strider.GetSink("output")->Subscribe();

  strider.Start();
  camera.Start();

  auto current = 0;
  for (auto i = 0; i < 5; i++) {
    auto id = reader->PopFrame()->GetValue<unsigned long>("frame_id");
    EXPECT_EQ(id, current);
    current += stride;
  }

  reader->UnSubscribe();
  camera.Stop();
  strider.Stop();
}
