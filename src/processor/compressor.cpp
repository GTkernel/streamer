
#include "processor/compressor.h"

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/thread/thread.hpp>

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

Compressor::Compressor(CompressionType t, unsigned int num_threads)
    : Processor(PROCESSOR_TYPE_COMPRESSOR, {SOURCE_NAME}, {SINK_NAME}),
      compression_type_(t),
      io_service_(),
      threadpool_(),
      work_(io_service_),
      stop_(false) {
  for (decltype(num_threads) i = 0; i < num_threads; ++i) {
    threadpool_.create_thread(
        boost::bind(&boost::asio::io_service::run, &io_service_));
  }
  io_service_.post(boost::bind(&Compressor::CommitAndPushFrames, this));
}

std::shared_ptr<Compressor> Compressor::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void Compressor::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr Compressor::GetSink() { return Processor::GetSink(SINK_NAME); }

bool Compressor::Init() { return true; }

bool Compressor::OnStop() {
  stop_ = true;
  io_service_.stop();
  threadpool_.join_all();
  return true;
}

void Compressor::Process() {
  // Get raw image
  auto frame = GetFrame(SOURCE_NAME);
  std::shared_ptr<LockedFrame> locked_frame =
      std::make_shared<LockedFrame>(std::move(frame));
  {
    std::lock_guard<std::mutex> lk(queue_lock_);
    queue_.emplace(locked_frame);
  }
  io_service_.post(boost::bind(&Compressor::DoCompression, this, locked_frame));
}

void Compressor::CommitAndPushFrames() {
  // Wait for the queue to have something in it
  while (true) {
    queue_lock_.lock();
    if (queue_.size() <= 0) {
      queue_lock_.unlock();
      continue;
    }
    auto ptr = queue_.front();
    queue_lock_.unlock();
    {
      std::unique_lock<std::mutex> lk(ptr->frame_lock);

      bool pred = false;
      // Loop until the wait return true, indicating that it was awoken before
      // the timeout.
      while (!pred) {
        pred = ptr->cv.wait_for(lk, std::chrono::milliseconds(100),
                                [ptr] { return ptr->count; });
        if (stop_) {
          // If the Compressor has been stopped, then we should return
          // immediately.
          return;
        }
      }

      queue_.pop();
    }
    PushFrame(SINK_NAME, std::move(ptr->frame));
  }
}

void Compressor::DoCompression(std::shared_ptr<LockedFrame> lf) {
  // Get raw image
  auto raw_image = lf->frame->GetValue<std::vector<char>>("original_bytes");

  // Compress raw
  std::vector<char> compressed_raw;
  boost::iostreams::filtering_ostream compressor;
  if (compression_type_ == CompressionType::BZIP2) {
    compressor.push(boost::iostreams::bzip2_compressor());
  } else if (compression_type_ == CompressionType::GZIP) {
    compressor.push(boost::iostreams::gzip_compressor());
  }
  compressor.push(boost::iostreams::back_inserter(compressed_raw));
  compressor.write((char*)raw_image.data(), raw_image.size());
  boost::iostreams::close(compressor);

  // Write compressed data to the frame and notify committer thread
  {
    std::lock_guard<std::mutex> lock(lf->frame_lock);
    lf->frame->SetValue("compressed_bytes", compressed_raw);
    lf->count = true;
  }
  lf->cv.notify_one();
}
