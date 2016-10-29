//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"

static const size_t SLIDING_WINDOW_SIZE = 25;

Processor::Processor() { Init_(); }

Processor::Processor(std::vector<std::shared_ptr<Stream>> sources,
                     std::vector<StreamPtr> sinks)
    : sources_(sources), sinks_(sinks) {
  Init_();
}

void Processor::Init_() {
  stopped_ = true;
  latency_sum_ = 0.0;
  sliding_latency_ = 99999.0;
  avg_latency_ = 0.0;
  n_processed_ = 0;
}

bool Processor::Start() {
  LOG(INFO) << "Start called";
  CHECK(stopped_) << "Processor has already started";
  stopped_ = false;
  process_thread_ = std::thread(&Processor::ProcessorLoop, this);
  return true;
}

bool Processor::Stop() {
  CHECK(!stopped_) << "Processor not started yet";
  stopped_ = true;
  process_thread_.join();
  bool result = OnStop();

  return result;
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  Timer timer;
  while (!stopped_) {
    // Cache source frames
    source_frame_cache_.clear();
    for (auto &stream : sources_) {
      source_frame_cache_.push_back(stream->PopFrame());
    }

    timer.Start();
    Process();
    n_processed_ += 1;
    double latency = timer.ElapsedMSec();
    {
      // Calculate latency
      latencies_.push(latency);
      latency_sum_ += latency;
      while (latencies_.size() > SLIDING_WINDOW_SIZE) {
        double oldest_latency = latencies_.front();
        latency_sum_ -= oldest_latency;
        latencies_.pop();
      }

      sliding_latency_ = latency_sum_ / latencies_.size();
    }
    avg_latency_ = (avg_latency_ * (n_processed_ - 1) + latency) / n_processed_;
  }
}

std::vector<std::shared_ptr<Stream>> Processor::GetSinks() { return sinks_; }

bool Processor::IsStarted() { return !stopped_; }

double Processor::GetSlidingLatencyMs() { return sliding_latency_; }

double Processor::GetAvgLatencyMs() { return avg_latency_; }

double Processor::GetAvgFps() { return 1000.0 / avg_latency_; }

template <typename FT>
std::shared_ptr<FT> Processor::GetFrame(int src_id) {
  CHECK(src_id < sources_.size());
  return std::dynamic_pointer_cast<FT>(source_frame_cache_[src_id]);
}

void Processor::PushFrame(int sink_id, Frame *frame) {
  CHECK(sink_id < sinks_.size());
  sinks_[sink_id]->PushFrame(frame);
}

template std::shared_ptr<Frame> Processor::GetFrame(int src_id);
template std::shared_ptr<ImageFrame> Processor::GetFrame(int src_id);
template std::shared_ptr<MetadataFrame> Processor::GetFrame(int src_id);
template std::shared_ptr<BytesFrame> Processor::GetFrame(int src_id);
