//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"

Processor::Processor() : stopped_(true) {}

Processor::Processor(std::vector<std::shared_ptr<Stream>> sources)
    : sources_(sources), stopped_(true) {}

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

  return OnStop();
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  while (!stopped_) {
    Process();
  }
}

std::vector<std::shared_ptr<Stream>> Processor::GetSinks() { return sinks_; }

bool Processor::Init() { return true; }
bool Processor::OnStop() { return true; }
