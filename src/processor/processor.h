//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef TX1DNN_PROCESSOR_H
#define TX1DNN_PROCESSOR_H

#include "common/common.h"
#include "stream/stream.h"
#include <thread>
/**
 * @brief Processor is the core computation unit in the system. It accepts frames
 * from one or more source streams, and output frames to one or more sink streams.
 */
class Processor {
 public:
  Processor();
  Processor(std::vector<std::shared_ptr<Stream>> sources);
  /**
   * @brief Start processing, drain frames from sources and send output to sinks.
   * @return True if start successfully, false otherwise.
   */
  bool Start();
  /**
   * @brief Stop processing
   * @return True if stop sucsessfully, false otherwise.
   */
  bool Stop();
 protected:
  /**
   * @brief Initialize the processor.
   */
  virtual bool Init();
  virtual bool OnStop();
  virtual void Consume() = 0;
  void ProcessorLoop();
  std::vector<std::shared_ptr<Stream>> sources_;
  std::vector<std::shared_ptr<Stream>> sinks_;
  std::thread process_thread_;
 private:
  bool stopped_;
};

#endif //TX1DNN_PROCESSOR_H
