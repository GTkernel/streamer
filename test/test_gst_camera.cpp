//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include <gtest/gtest.h>
#include "camera/gst_camera.h"

TEST(GST_CAMERA_TEST, TEST_BASIC) {
  std::string camera_name = "TEST_CAMERA";
  std::string video_uri =
      "gst://videotestsrc ! video/x-raw,width=640,height=480";
  int width = 640;
  int height = 480;
  GSTCamera camera(camera_name, video_uri, 640, 480);

  auto stream = camera.GetStream();
  auto reader = stream->Subscribe();
  camera.Start();

  const auto& image = reader->PopFrame()->GetValue<cv::Mat>("original_image");

  EXPECT_EQ(height, image.rows);
  EXPECT_EQ(width, image.cols);

  reader->UnSubscribe();
  camera.Stop();
}

TEST(GST_CAMERA_TEST, TEST_CAPTURE) {
  std::string camera_name = "TEST_CAMERA";
  std::string video_uri =
      "gst://videotestsrc ! video/x-raw,width=640,height=480";
  int width = 640;
  int height = 480;
  std::shared_ptr<Camera> camera(
      new GSTCamera(camera_name, video_uri, 640, 480));

  // Can capture image when camera is not started
  cv::Mat image;
  bool result = camera->Capture(image);
  EXPECT_EQ(true, result);
  EXPECT_EQ(height, image.rows);
  EXPECT_EQ(width, image.cols);

  // Can also capture image when camera is started
  camera->Start();
  result = camera->Capture(image);
  EXPECT_EQ(true, result);
  EXPECT_EQ(height, image.rows);
  EXPECT_EQ(width, image.cols);
  camera->Stop();
}
