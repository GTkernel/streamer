//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_COMMON_H
#define TX1_DNN_COMMON_H

#include <opencv2/opencv.hpp>
#include <string>
#include <glog/logging.h>
#include <thread>
#include <memory>
#include <stdlib.h>
#include <tinytoml/toml.h>
#include <fstream>
#include "timer.h"
#include "types.h"

using std::string;

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

//// TOML
toml::Value ParseTomlFromFile(const string &filepath);

//// Global context. Singleton class.
class Context {
 public:
  static Context &GetContext();
 public:
  Context();
  string GetConfigDir() const;
  string GetConfigFile(const string &filename) const;
  void SetConfigDir(const string &config_dir);
 private:
  string config_dir_;
};

#endif //TX1_DNN_COMMON_H
