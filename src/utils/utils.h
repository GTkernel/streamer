//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_UTILS_UTILS_H_
#define STREAMER_UTILS_UTILS_H_

#include "utils/cuda_utils.h"
#include "utils/cv_utils.h"
#include "utils/file_utils.h"
#include "utils/gst_utils.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/time_utils.h"
#include "utils/yolo_utils.h"

#include <sched.h>
#include <condition_variable>
#include <queue>
#include <thread>

#define STREAMER_NOT_IMPLEMENTED (LOG(FATAL) << "Function not implemented");

inline void STREAMER_SLEEP(int msecs) {
  std::this_thread::sleep_for(std::chrono::milliseconds(msecs));
}

/**
 * @brief A thread safe queue. Pop items from the an empty queue will block
 * until an item is available.
 */
template <typename T>
class TaskQueue {
 public:
  void Push(const T& t) {
    std::lock_guard<std::mutex> lock(queue_lock_);
    queue_.push(t);
    queue_cv_.notify_all();
  }

  T Pop() {
    std::unique_lock<std::mutex> lk(queue_lock_);
    queue_cv_.wait(lk, [this] { return queue_.size() != 0; });
    T t = queue_.front();
    queue_.pop();

    return t;
  }

 private:
  std::queue<T> queue_;
  std::mutex queue_lock_;
  std::condition_variable queue_cv_;
};

//// TOML
inline toml::Value ParseTomlFromFile(const std::string& filepath) {
  std::ifstream ifs(filepath);
  CHECK(!ifs.fail()) << "Can't open file " << filepath << " for read";
  toml::ParseResult pr = toml::parse(ifs);
  CHECK(pr.valid()) << "Toml file " << filepath
                    << " is not a valid toml file:" << std::endl
                    << pr.errorReason;
  return pr.value;
}

inline void ExecuteAndCheck(std::string command) {
  int exit_code = system(command.c_str());
  if (!exit_code) {
    LOG(FATAL) << "Command \"" << command
               << "\"failed with exit code: " << exit_code;
  }
}

#endif  // STREAMER_UTILS_UTILS_H_
