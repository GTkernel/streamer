//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef TX1DNN_PROCESSOR_H
#define TX1DNN_PROCESSOR_H

#include <atomic>
#include <thread>
#include "common/common.h"
#include "stream/stream.h"
/**
 * @brief Processor is the core computation unit in the system. It accepts
 * frames
 * from one or more source streams, and output frames to one or more sink
 * streams.
 */
class Processor {
 public:
  Processor();
  Processor(std::vector<StreamPtr> sources, std::vector<StreamPtr> sinks);
  /**
   * @brief Start processing, drain frames from sources and send output to
   * sinks.
   * @return True if start successfully, false otherwise.
   */
  bool Start();
  /**
   * @brief Stop processing
   * @return True if stop sucsessfully, false otherwise.
   */
  bool Stop();
  /**
   * @brief Get sink streams of the prosessor.
   */
  std::vector<std::shared_ptr<Stream>> GetSinks();

  /**
   * @brief Check if the processor has started.
   * @return True if processor has started.
   */
  bool IsStarted();

  /**
   * @brief Get sliding window average latency of the processor.
   * @return Latency in ms
   */
  double GetLatencyMs();

  /**
   * @brief Get processing speed of the processor, measured in frames / sec. It
   * is simply computed as 1000.0 / GetLatencyMs().
   * @return FPS of the processor.
   */
  double GetFps();

 protected:
  /**
   * @brief Initialize the processor.
   */
  virtual bool Init() = 0;
  virtual bool OnStop() = 0;
  virtual void Process() = 0;
  void ProcessorLoop();
  std::vector<std::shared_ptr<Stream>> sources_;
  std::vector<std::shared_ptr<Stream>> sinks_;
  std::thread process_thread_;
  bool stopped_;

  // Process latency, sliding window average of 10 samples;
  std::queue<double> latencies_;
  double latency_sum_;
  std::atomic<double> latency_;
};

#endif  // TX1DNN_PROCESSOR_H
