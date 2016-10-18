//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"

static const size_t SLIDING_WINDOW_SIZE = 10;

Processor::Processor() : stopped_(true) {}

Processor::Processor(std::vector<std::shared_ptr<Stream>> sources,
                     std::vector<StreamPtr> sinks)
    : sources_(sources),
      sinks_(sinks),
      stopped_(true),
      latency_sum_(0.0),
      latency_(999999.0) {}

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
    timer.Start();
    Process();
    double latency = timer.ElapsedMSec();
    latencies_.push(latency);
    latency_sum_ += latency;
    while (latencies_.size() > SLIDING_WINDOW_SIZE) {
      double oldest_latency = latencies_.front();
      latency_sum_ -= oldest_latency;
      latencies_.pop();
    }
    latency_ = latency_sum_ / SLIDING_WINDOW_SIZE;
  }
}

std::vector<std::shared_ptr<Stream>> Processor::GetSinks() { return sinks_; }

bool Processor::IsStarted() { return !stopped_; }

double Processor::GetLatencyMs() { return latency_; }

double Processor::GetFps() { return 1000.0 / latency_; }
