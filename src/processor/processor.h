//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_PROCESSOR_H
#define STREAMER_PROCESSOR_H

#include <atomic>
#include <thread>
#include "common/common.h"
#include "stream/stream.h"

class Pipeline;

#ifdef USE_VIMBA
class VimbaCameraFrameObserver;
#endif

/**
 * @brief Processor is the core computation unit in the system. It accepts
 * frames from one or more source streams, and output frames to one or more sink
 * streams.
 */
class Processor {
  friend class Pipeline;
#ifdef USE_VIMBA
  friend class VimbaCameeraFrameObserver;
#endif
 public:
  Processor(const std::vector<string> &source_names,
            const std::vector<string> &sink_names);
  virtual ~Processor();
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
   * @brief Get sink stream of the processor by name.
   * @param name Name of the sink.
   * @return Stream with the name.
   */
  StreamPtr GetSink(const string &name);

  /**
   * @brief Set the source of the processor by name.
   * @param name Name of the source.
   * @param stream Stream to be set.
   */
  void SetSource(const string &name, StreamPtr stream);

  /**
   * @brief Check if the processor has started.
   * @return True if processor has started.
   */
  bool IsStarted();

  /**
   * @brief Get sliding window average latency of the processor.
   * @return Latency in ms.
   */
  double GetSlidingLatencyMs();

  /**
   * @brief Get overall average latency.
   * @return Latency in ms.
   */
  double GetAvgLatencyMs();

  /**
   * @brief Get processing speed of the processor, measured in frames / sec. It
   * is simply computed as 1000.0 / GetLatencyMs().
   * @return FPS of the processor.
   */
  double GetAvgFps();

  /**
   * @brief Get the type of the processor
   */
  virtual ProcessorType GetType() = 0;

 protected:
  /**
   * @brief Initialize the processor.
   */
  virtual bool Init() = 0;
  /**
   * @brief Called after Prcessor#Stop() is called, do any clean up in this
   * method.
   * @return [description]
   */
  virtual bool OnStop() = 0;
  /**
   * @brief Fetch one frame from sources_ and process them.
   *
   * @input_frames A list of input frames feteched from sources_.
   * @return A list of output frames.
   */
  virtual void Process() = 0;

  template <typename FT = Frame>
  std::shared_ptr<FT> GetFrame(const string &source_name);
  void PushFrame(const string &sink_name, std::shared_ptr<Frame> frame);
  void PushFrame(const string &sink_name, Frame *frame);
  void Init_();

  void ProcessorLoop();

  std::unordered_map<string, std::shared_ptr<Frame>> source_frame_cache_;
  std::unordered_map<string, StreamPtr> sources_;
  std::unordered_map<string, StreamPtr> sinks_;
  std::unordered_map<string, StreamReader *> readers_;

  std::thread process_thread_;
  bool stopped_;

  // Process latency, sliding window average of 10 samples;
  std::queue<double> latencies_;
  double latency_sum_;
  double sliding_latency_;
  double avg_latency_;

  // Processor stats
  // Number of processed frames
  size_t n_processed_;
};

#endif  // STREAMER_PROCESSOR_H
