//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"
#include "utils/utils.h"

static const size_t SLIDING_WINDOW_SIZE = 25;

Processor::Processor(ProcessorType type,
                     const std::vector<string>& source_names,
                     const std::vector<string>& sink_names)
    : num_frames_processed_(0),
      avg_processing_latency_ms_(0),
      processing_latencies_sum_ms_(0),
      trailing_avg_processing_latency_ms_(0),
      queue_latency_sum_ms_(0),
      type_(type),
      block_on_push_(false) {
  for (const auto& source_name : source_names) {
    sources_.insert({source_name, nullptr});
    source_frame_cache_[source_name] = nullptr;
  }

  for (const auto& sink_name : sink_names) {
    sinks_.insert({sink_name, StreamPtr(new Stream)});
  }

  Init_();

  control_socket_ =
      new zmq::socket_t(*Context::GetContext().GetControlContext(), ZMQ_PUSH);
  // XXX: PUSH sockets can only send
  control_socket_->connect(Context::GetControlChannelName());
  int linger = 0;
  control_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
}

Processor::~Processor() {
  control_socket_->close();
  delete control_socket_;
}

StreamPtr Processor::GetSink(const string& name) { return sinks_[name]; }

void Processor::SetSource(const string& name, StreamPtr stream) {
  if (sources_.find(name) == sources_.end()) {
    std::ostringstream sources;
    for (const auto& s : sources_) {
      sources << "`" << s.first << "' ";
    }
    LOG(FATAL) << "Source `" << name << "` does not exist.\n"
               << "Available sources: " << sources.str();
  }
  sources_[name] = stream;
}

void Processor::Init_() { stopped_ = true; }

bool Processor::Start() {
  LOG(INFO) << "Start called";
  CHECK(stopped_) << "Processor has already started";

  // Check sources are filled
  for (const auto& source : sources_) {
    CHECK(source.second != nullptr)
        << "Source \"" << source.first << "\" is not set.";
  }

  // Subscribe sources
  for (auto& source : sources_) {
    readers_.emplace(source.first, source.second->Subscribe());
  }

  stopped_ = false;
  process_thread_ = std::thread(&Processor::ProcessorLoop, this);
  return true;
}

bool Processor::Stop() {
  LOG(INFO) << "Stop called";
  CHECK(!stopped_) << "Processor not started yet";

  // This signals that the process thread should stop processing new frames.
  stopped_ = true;

  // Stop all sink streams, which wakes up any blocking calls to
  // Stream::PushFrame() in the process thread.
  for (const auto& p : sinks_) {
    p.second->Stop();
  }

  // Join the process thread, completing the main processing loop.
  process_thread_.join();

  // Do any processor-specific cleanup.
  bool result = OnStop();

  // Unsubscribe from the source streams.
  for (const auto& reader : readers_) {
    reader.second->UnSubscribe();
  }

  // Deallocate the source StreamReaders.
  readers_.clear();

  return result;
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  Timer local_timer;
  while (!stopped_) {
    // Cache source frames
    source_frame_cache_.clear();
    for (auto& reader : readers_) {
      auto source_name = reader.first;
      auto source_stream = reader.second;

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
          // Calculate queue latency
          double start_ms = frame->GetValue<double>("start_time_ms");
          double end_ms = Context::GetContext().GetTimer().ElapsedMSec();
          queue_latency_sum_ms_ += end_ms - start_ms;
          source_frame_cache_[source_name] = std::move(frame);
          break;
        }
      }
    }

    local_timer.Start();
    Process();
    ++num_frames_processed_;
    double processing_latency_ms = local_timer.ElapsedMSec();

    // Update average processing latency.
    avg_processing_latency_ms_ =
        (avg_processing_latency_ms_ * (num_frames_processed_ - 1) +
         processing_latency_ms) /
        num_frames_processed_;

    // Update trailing average processing latency.
    auto num_latencies = processing_latencies_ms_.size();
    if (num_latencies == SLIDING_WINDOW_SIZE) {
      // Pop the oldest latency from the queue and remove it from the running
      // sum.
      double oldest_latency_ms = processing_latencies_ms_.front();
      processing_latencies_ms_.pop();
      processing_latencies_sum_ms_ -= oldest_latency_ms;
    }
    // Add the new latency to the queue and the running sum.
    processing_latencies_ms_.push(processing_latency_ms);
    processing_latencies_sum_ms_ += processing_latency_ms;
    trailing_avg_processing_latency_ms_ =
        processing_latencies_sum_ms_ / num_latencies;
  }
}

bool Processor::IsStarted() const { return !stopped_; }

double Processor::GetTrailingAvgProcessingLatencyMs() const {
  return trailing_avg_processing_latency_ms_;
}

double Processor::GetAvgProcessingLatencyMs() const {
  return avg_processing_latency_ms_;
}

double Processor::GetAvgQueueLatencyMs() const {
  return queue_latency_sum_ms_ / num_frames_processed_;
}

ProcessorType Processor::GetType() const { return type_; }

zmq::socket_t* Processor::GetControlSocket() { return control_socket_; };

void Processor::SetBlockOnPush(bool block) { block_on_push_ = block; }

void Processor::PushFrame(const string& sink_name,
                          std::unique_ptr<Frame> frame) {
  CHECK(sinks_.count(sink_name) != 0);
  sinks_[sink_name]->PushFrame(std::move(frame), block_on_push_);
}

std::unique_ptr<Frame> Processor::GetFrame(const string& source_name) {
  CHECK(source_frame_cache_.count(source_name) != 0);
  return std::move(source_frame_cache_[source_name]);
}
