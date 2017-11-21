//
// Created by Ran Xian (xranthoar@gmail.com) on 9/25/16.
//

#include <stdio.h>

#include <gst/gst.h>
#include <gtest/gtest.h>
#include "common/context.h"

GTEST_API_ int main(int argc, char** argv) {
  printf("Running main() from streamer_gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;
  if (argc >= 2) {
    std::string config_dir = argv[1];
    LOG(INFO) << config_dir;
    Context::GetContext().SetConfigDir(config_dir);
  } else {
    // Try to guess the test config file
    if (FileExists("./test/config/cameras.toml")) {
      LOG(INFO) << "Use config from ./test/config";
      Context::GetContext().SetConfigDir("./test/config");
    } else if (FileExists("./config/cameras.toml")) {
      LOG(INFO) << "Use config from ./config";
      Context::GetContext().SetConfigDir("./config");
    }
  }

  Context::GetContext().Init();
  return RUN_ALL_TESTS();
}
