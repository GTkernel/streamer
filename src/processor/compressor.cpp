#include "compressor.h"

#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

Compressor::Compressor(Compressor::CompressionType t, int num_threads)
    : Processor(PROCESSOR_TYPE_CUSTOM, {"input"}, {}),
      compression_type_(t),
      ioService_(),
      threadpool_(),
      work_(ioService_) {
  for (int i = 0; i < num_threads; ++i) {
    threadpool_.create_thread(
        boost::bind(&boost::asio::io_service::run, &ioService_));
  }
  ioService_.post(boost::bind(&Compressor::CommitAndPushFrames, this));
  sinks_.insert({"output", std::make_shared<Stream>("output")});
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
      std::unique_lock<std::mutex> lk(ptr->lock);
      while (ptr->count < 1) {
        ptr->cv.wait(lk);
      }
      queue_.pop();
    }
    PushFrame("output", std::move(ptr->frame));
  }
}

bool Compressor::Init() { return true; }

bool Compressor::OnStop() {
  ioService_.stop();
  threadpool_.join_all();
  return true;
}

void Compressor::DoCompression(std::shared_ptr<Compressor::LockedFrame> lf) {
  // Get raw image
  auto raw_image = lf->frame->GetValue<std::vector<char>>("original_bytes");

  // Compress raw
  std::vector<char> compressed_raw;
  boost::iostreams::filtering_ostream compressor;
  if (compression_type_ == Compressor::CompressionType::BZIP2) {
    compressor.push(boost::iostreams::bzip2_compressor());
  } else if (compression_type_ == Compressor::CompressionType::GZIP) {
    compressor.push(boost::iostreams::gzip_compressor());
  }
  compressor.push(boost::iostreams::back_inserter(compressed_raw));
  compressor.write((char*)raw_image.data(), raw_image.size());
  boost::iostreams::close(compressor);

  // Write compressed data to the frame and notify committer thread
  {
    std::lock_guard<std::mutex>(lf->lock);
    lf->frame->SetValue("compressed_bytes", compressed_raw);
    lf->count = 1;
  }
  lf->cv.notify_one();
}

void Compressor::Process() {
  // Get raw image
  auto frame = GetFrame("input");
  std::shared_ptr<LockedFrame> locked_frame =
      std::make_shared<LockedFrame>(std::move(frame));
  {
    std::lock_guard<std::mutex> lk(queue_lock_);
    queue_.emplace(locked_frame);
  }
  ioService_.post(boost::bind(&Compressor::DoCompression, this, locked_frame));
}
