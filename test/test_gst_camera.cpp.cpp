//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include <gtest/gtest.h>
#include "camera/gst_camera.h"

TEST(GST_CAMERA_TEST, TEST_BASIC) {
  string camera_name = "TEST_CAMERA";
  string video_uri = "gst://videotestsrc ! video/x-raw,width=640,height=480";
  int width = 640;
  int height = 480;
  GSTCamera camera(camera_name, video_uri, 640, 480);

  auto stream = camera.GetStream();
  camera.Start();

  cv::Mat image = stream->PopImageFrame()->GetImage();
  camera.Stop();
  EXPECT_EQ(image.rows, height);
  EXPECT_EQ(image.cols, width);
}