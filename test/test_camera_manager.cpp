//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include <gtest/gtest.h>
#include "camera/camera_manager.h"

TEST(CAMERA_MANAGER_TEST, TEST_BASIC) {
  CameraManager &manager = CameraManager::GetInstance();

  EXPECT_EQ(manager.GetCameras().size(), 2);
  EXPECT_EQ(manager.GetCamera("testcam2")->GetVideoURI(), "facetime");
}