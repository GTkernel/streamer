//
// Created by Ran Xian (xranthoar@gmail.com) on 10/26/16.
//

#include <gtest/gtest.h>
#include <streamer.h>

TEST(GST_VIDEO_ENCODER_TEST, FILE_TEST) {
  CameraManager& camera_manager = CameraManager::GetInstance();

  auto camera = camera_manager.GetCamera("GST_TEST");
  GstVideoEncoder encoder(camera->GetWidth(), camera->GetHeight(), "test.mp4");
  encoder.SetSource("input", camera->GetStream());
  camera->Start();
  encoder.Start();

  STREAMER_SLEEP(100);

  auto encoder_reader = encoder.GetSink("output")->Subscribe();
  auto image_frame = encoder_reader->PopFrame();

  encoder.Stop();
  camera->Stop();

  std::remove("test.mp4");
}

TEST(GST_VIDEO_ENCODER_TEST, STREAM_TEST) {
  CameraManager& camera_manager = CameraManager::GetInstance();

  auto camera = camera_manager.GetCamera("GST_TEST");
  GstVideoEncoder encoder(camera->GetWidth(), camera->GetHeight(), 12345);
  encoder.SetSource("input", camera->GetStream());
  camera->Start();
  encoder.Start();

  STREAMER_SLEEP(100);

  encoder.Stop();
  camera->Stop();
}
