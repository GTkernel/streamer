//
// Created by Ran Xian (xranthoar@gmail.com) on 9/25/16.
//

#include <stdio.h>

#include <gst/gst.h>
#include <gtest/gtest.h>
#include "common/common.h"
#include "common/context.h"

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from tx1dnn_gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;
  if (argc >= 2) {
    string config_dir = argv[1];
    LOG(INFO) << config_dir;
    Context::GetContext().SetConfigDir(config_dir);
    Context::GetContext().Init();
  }
  return RUN_ALL_TESTS();
}
