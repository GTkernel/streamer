#ifndef STREAMER_PROCESSOR_COMPRESSOR_H_
#define STREAMER_PROCESSOR_COMPRESSOR_H_

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#include "common/types.h"
#include "processor/processor.h"

class Compressor : public Processor {
 public:
  enum CompressionType { BZIP2, GZIP, NONE };

  Compressor(CompressionType t, unsigned int num_threads = 4);
  static std::shared_ptr<Compressor> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  struct LockedFrame {
   public:
    LockedFrame(std::unique_ptr<Frame> frame_ptr) : count(0) {
      frame = std::move(frame_ptr);
    }

    std::unique_ptr<Frame> frame;
    bool count;
    std::mutex frame_lock;
    std::condition_variable cv;
  };

  void CommitAndPushFrames();
  void DoCompression(std::shared_ptr<Compressor::LockedFrame> lf);

  CompressionType compression_type_;
  boost::asio::io_service io_service_;
  boost::thread_group threadpool_;
  boost::asio::io_service::work work_;
  std::mutex queue_lock_;
  std::queue<std::shared_ptr<LockedFrame>> queue_;
  // Used to signal the wait loop in "CommitAndPushFrames()" that it is time to
  // stop.
  bool stop_;
};

#endif  // STREAMER_PROCESSOR_COMPRESSOR_H_
