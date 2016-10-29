//
// Created by Ran Xian (xranthoar@gmail.com) on 10/26/16.
//

#include <gtest/gtest.h>
#include <streamer.h>

TEST(GST_VIDEO_ENCODER_TEST, TEST_BASIC) {
  CameraManager &camera_manager = CameraManager::GetInstance();

  auto camera = camera_manager.GetCamera("GST_TEST");
  GstVideoEncoder encoder(camera->GetStream(), camera->GetWidth(),
                          camera->GetHeight(), "test.mp4");
  camera->Start();
  encoder.Start();

  STREAMER_SLEEP(0.1);

  encoder.Stop();
  camera->Stop();

  std::remove("test.mp4");
}