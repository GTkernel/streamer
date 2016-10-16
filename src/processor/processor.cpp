//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"

Processor::Processor() : stopped_(true) {}

Processor::Processor(std::vector<std::shared_ptr<Stream>> sources, std::vector<StreamPtr> sinks)
    : sources_(sources), sinks_(sinks), stopped_(true) {}

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
  bool result = OnStop();
  process_thread_.join();

  return result;
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  while (!stopped_) {
    Process();
  }
}

std::vector<std::shared_ptr<Stream>> Processor::GetSinks() { return sinks_; }

bool Processor::IsStarted() {
  return !stopped_;
}
