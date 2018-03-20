#ifndef STREAMER_PROCESSOR_COMPRESSOR_H_
#define STREAMER_PROCESSOR_COMPRESSOR_H_

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>

#include "common/types.h"
#include "processor/processor.h"

class Compressor : public Processor {
 public:
  enum CompressionType { BZIP2, GZIP, NONE };

  Compressor(CompressionType t);
  ~Compressor();
  static std::shared_ptr<Compressor> Create(const FactoryParamsType& params);
  static std::string CompressionTypeToString(CompressionType type);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

  static const char* kDataKey;
  static const char* kTypeKey;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  void OutputFrames();
  std::unique_ptr<Frame> CompressFrame(std::unique_ptr<Frame>);

  CompressionType compression_type_;
  std::queue<std::future<std::unique_ptr<Frame>>> queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cond_;
  std::thread output_thread_;
  std::atomic<bool> stop_;
};

#endif  // STREAMER_PROCESSOR_COMPRESSOR_H_
