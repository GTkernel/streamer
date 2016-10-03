//
// Created by Ran Xian (xranthoar@gmail.com) on 9/25/16.
//

#include <gtest/gtest.h>
#include "model/model_manager.h"

TEST(CAMERA_MANAGER_TEST, TEST_BASIC) {
  ModelManager &manager = ModelManager::GetInstance();

  auto mean_colors = manager.GetMeanColors();
  EXPECT_EQ(mean_colors[0], 104);
  EXPECT_EQ(mean_colors[1], 117);
  EXPECT_EQ(mean_colors[2], 123);
  EXPECT_EQ(manager.GetModelDescs().size(), 2);
  EXPECT_EQ(manager.GetModelDesc("AlexNet").GetModelDescPath(),
            "/path/to/caffe/model.prototxt");
}