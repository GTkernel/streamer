
#include <memory>

#include <gtest/gtest.h>

#include "camera/camera_manager.h"
#include "stream/frame.h"
#include "utils/utils.h"
#include "video/gst_video_encoder.h"

TEST(TestGstVideoEncoder, TestFile) {
  auto camera = CameraManager::GetInstance().GetCamera("GST_TEST");
  auto encoder =
      std::make_shared<GstVideoEncoder>("original_image", 30, "/tmp/test.mp4");
  encoder->SetSource(camera->GetStream());

  auto encoder_reader = encoder->GetSink()->Subscribe();
  camera->Start();
  encoder->Start();

  auto image_frame = encoder_reader->PopFrame();

  encoder_reader->UnSubscribe();
  encoder->Stop();
  camera->Stop();

  std::remove("test.mp4");
}

TEST(TestGstVideoEncoder, TestStream) {
  auto camera = CameraManager::GetInstance().GetCamera("GST_TEST");
  auto encoder = std::make_shared<GstVideoEncoder>("original_image", 30, 12345);
  encoder->SetSource(camera->GetStream());

  camera->Start();
  encoder->Start();

  STREAMER_SLEEP(100);

  encoder->Stop();
  camera->Stop();
}
