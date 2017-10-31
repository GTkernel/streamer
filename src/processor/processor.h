//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_PROCESSOR_PROCESSOR_H_
#define STREAMER_PROCESSOR_PROCESSOR_H_

#include <atomic>
#include <queue>
#include <thread>
#include <unordered_map>

#include <zmq.hpp>

#include "common/common.h"
#include "stream/stream.h"

class Pipeline;

#ifdef USE_VIMBA
class VimbaCameraFrameObserver;
#endif  // USE_VIMBA

/**
 * @brief Processor is the core computation unit in the system. It accepts
 * frames from one or more source streams, and output frames to one or more sink
 * streams.
 */
class Processor {
  friend class Pipeline;
#ifdef USE_VIMBA
  friend class VimbaCameeraFrameObserver;
#endif  // USE_VIMBA
 public:
  Processor(ProcessorType type, const std::vector<string>& source_names,
            const std::vector<string>& sink_names);
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
  StreamPtr GetSink(const string& name);

  /**
   * @brief Set the source of the processor by name.
   * @param name Name of the source.
   * @param stream Stream to be set.
   */
  virtual void SetSource(const string& name, StreamPtr stream);

  /**
   * @brief Check if the processor has started.
   * @return True if processor has started.
   */
  bool IsStarted() const;

  /**
   * @brief Get trailing (sliding window) average latency of the processor.
   * @return Latency in ms.
   */
  virtual double GetTrailingAvgProcessingLatencyMs() const;

  /**
   * @brief Get overall average latency.
   * @return Latency in ms.
   */
  virtual double GetAvgProcessingLatencyMs() const;

  /**
   * @brief Get overall average queue latency.
   * @return queue latency in ms.
   */
  virtual double GetAvgQueueLatencyMs() const;

  /**
   * @brief Get overall throughput of processor.
   * @return in FPS.
   */
  virtual double GetHistoricalProcessFps();

  /**
   * @brief Get the type of the processor
   */
  ProcessorType GetType() const;

  zmq::socket_t* GetControlSocket();

  // Configure whether this processor should block when pushing frames to its
  // outputs streams if any of its output streams is full.
  virtual void SetBlockOnPush(bool block);

 protected:
  /**
   * @brief Initialize the processor.
   */
  virtual bool Init() = 0;
  /**
   * @brief Called after Processor#Stop() is called, do any clean up in this
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

  std::unique_ptr<Frame> GetFrame(const string& source_name);
  void PushFrame(const string& sink_name, std::unique_ptr<Frame> frame);
  void ProcessorLoop();

  std::unordered_map<string, std::unique_ptr<Frame>> source_frame_cache_;
  std::unordered_map<string, StreamPtr> sources_;
  std::unordered_map<string, StreamPtr> sinks_;
  std::unordered_map<string, StreamReader*> readers_;

  std::thread process_thread_;
  std::atomic<bool> stopped_;
  std::atomic<bool> found_last_frame_;

  unsigned int num_frames_processed_;
  double avg_processing_latency_ms_;
  // A queue of recent processing latencies.
  std::queue<double> processing_latencies_ms_;
  // The sum of all of the values in "processing_latencies_ms_". This is purely
  // a performance optimization to avoid needing to sum over
  // "processing_latencies_ms_" for every frame.
  double processing_latencies_sum_ms_;
  // Processing latency, computed using a sliding window average.
  double trailing_avg_processing_latency_ms_;
  double queue_latency_sum_ms_;

 private:
  const ProcessorType type_;
  zmq::socket_t* control_socket_;
  Timer processor_timer_;
  // Whether to block when pushing frames if any output streams are full.
  std::atomic<bool> block_on_push_;
};

#endif  // STREAMER_PROCESSOR_PROCESSOR_H_
