#ifndef STREAMER_PROCESSOR_COMPRESSOR_H_
#define STREAMER_PROCESSOR_COMPRESSOR_H_

#include <stdlib.h>
#include <chrono>
#include <fstream>

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#include "common/types.h"
#include "processor/processor.h"

class Compressor : public Processor {
 public:
  enum CompressionType { BZIP2, GZIP, NONE };

  struct LockedFrame {
   public:
    LockedFrame(std::unique_ptr<Frame> frame_ptr) : count(0) {
      frame = std::move(frame_ptr);
    }
    std::unique_ptr<Frame> frame;
    int count;
    std::mutex lock;
    std::condition_variable cv;
  };

  Compressor(CompressionType t, int num_threads = 4);
  static std::shared_ptr<Compressor> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  void CommitAndPushFrames();
  Compressor::CompressionType compression_type_;
  void DoCompression(std::shared_ptr<Compressor::LockedFrame> lf);

  boost::asio::io_service ioService_;
  boost::thread_group threadpool_;
  boost::asio::io_service::work work_;
  std::mutex queue_lock_;
  std::queue<std::shared_ptr<LockedFrame>> queue_;
  std::mutex global_lock_;
};

#endif  // STREAMER_PROCESSOR_COMPRESSOR_H_
