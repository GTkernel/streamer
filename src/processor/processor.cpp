//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor.h"
#include "utils/utils.h"

static const size_t SLIDING_WINDOW_SIZE = 25;

Processor::Processor(ProcessorType type,
                     const std::vector<string>& source_names,
                     const std::vector<string>& sink_names)
    : type_(type) {
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

void Processor::Init_() {
  stopped_ = true;
  latency_sum_ = 0.0;
  sliding_latency_ = 99999.0;
  avg_latency_ = 0.0;
  queue_latency_sum_ = 0.0;
  n_processed_ = 0;
}

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

  timer_.Start();

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

  for (auto& reader : readers_) {
    reader.second->UnSubscribe();
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
          double start = frame->GetValue<double>("start_time_ms");
          double end = Context::GetContext().GetTimer().ElapsedMSec();
          source_frame_cache_[source_name] = std::move(frame);
          queue_latency_sum_ += end - start;
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

bool Processor::IsStarted() const { return !stopped_; }

double Processor::GetSlidingLatencyMs() const { return sliding_latency_; }

double Processor::GetAvgLatencyMs() const { return avg_latency_; }

double Processor::GetAvgQueueLatencyMs() const {
  return queue_latency_sum_ / n_processed_;
}

double Processor::GetObservedAvgFps() {
  double secs_elapsed = timer_.ElapsedMSec() / 1000;
  return n_processed_ / secs_elapsed;
}

double Processor::GetAvgFps() const { return 1000.0 / avg_latency_; }

ProcessorType Processor::GetType() const { return type_; }

zmq::socket_t* Processor::GetControlSocket() { return control_socket_; };

void Processor::PushFrame(const string& sink_name,
                          std::unique_ptr<Frame> frame) {
  CHECK(sinks_.count(sink_name) != 0);
  sinks_[sink_name]->PushFrame(std::move(frame));
}

std::unique_ptr<Frame> Processor::GetFrame(const string& source_name) {
  CHECK(source_frame_cache_.count(source_name) != 0);
  return std::move(source_frame_cache_[source_name]);
}
