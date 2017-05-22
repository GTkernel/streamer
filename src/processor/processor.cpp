//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"
#include "utils/utils.h"

static const size_t SLIDING_WINDOW_SIZE = 25;

Processor::Processor(const std::vector<string> &source_names,
                     const std::vector<string> &sink_names) {
  for (auto &source_name : source_names) {
    sources_.insert({source_name, nullptr});
    source_frame_cache_.insert({source_name, nullptr});
  }

  for (auto &sink_name : sink_names) {
    sinks_.insert({sink_name, StreamPtr(new Stream)});
  }

  Init_();
}

Processor::~Processor() {}

StreamPtr Processor::GetSink(const string &name) { return sinks_[name]; }

void Processor::SetSource(const string &name, StreamPtr stream) {
  sources_[name] = stream;
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

  // Check sources are filled
  for (auto itr = sources_.begin(); itr != sources_.end(); itr++) {
    CHECK(itr->second != nullptr) << "Source: " << itr->first << " is not set.";
  }

  // Subscribe sources
  for (auto itr = sources_.begin(); itr != sources_.end(); itr++) {
    readers_.emplace(itr->first, itr->second->Subscribe());
  }

  stopped_ = false;
  process_thread_ = std::thread(&Processor::ProcessorLoop, this);
  return true;
}

bool Processor::Stop() {
  CHECK(!stopped_) << "Processor not started yet";
  stopped_ = true;
  LOG(INFO) << "Stop called";
  process_thread_.join();
  bool result = OnStop();

  for (auto itr = readers_.begin(); itr != readers_.end(); itr++) {
    itr->second->UnSubscribe();
  }

  readers_.clear();

  return result;
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  Timer timer;
  while (!stopped_) {
    // Cache source frames
    source_frame_cache_.clear();
    for (auto itr = readers_.begin(); itr != readers_.end(); itr++) {
      auto source_name = itr->first;
      auto source_stream = itr->second;

      while (true) {
        auto frame = source_stream->PopFrame(100);
        if (frame == nullptr) {
          if (stopped_) {
            // We can't get frame, we should have stopped
            return;
          } else {
            continue;
          }
        } else {
          source_frame_cache_.insert({source_name, frame});
          break;
        }
      }
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

bool Processor::IsStarted() { return !stopped_; }

double Processor::GetSlidingLatencyMs() { return sliding_latency_; }

double Processor::GetAvgLatencyMs() { return avg_latency_; }

double Processor::GetAvgFps() { return 1000.0 / avg_latency_; }

void Processor::PushFrame(const string &sink_name, std::shared_ptr<Frame> frame) {
  CHECK(sinks_.count(sink_name) != 0);
  sinks_[sink_name]->PushFrame(frame);
}

void Processor::PushFrame(const string &sink_name, Frame *frame) {
  CHECK(sinks_.count(sink_name) != 0);
  sinks_[sink_name]->PushFrame(frame);
}

template <typename FT>
std::shared_ptr<FT> Processor::GetFrame(const string &source_name) {
  CHECK(source_frame_cache_.count(source_name) != 0);
  return std::dynamic_pointer_cast<FT>(source_frame_cache_[source_name]);
}

template std::shared_ptr<Frame> Processor::GetFrame(const string &source_name);
template std::shared_ptr<ImageFrame> Processor::GetFrame(
    const string &source_name);
template std::shared_ptr<MetadataFrame> Processor::GetFrame(
    const string &source_name);
template std::shared_ptr<BytesFrame> Processor::GetFrame(
    const string &source_name);
