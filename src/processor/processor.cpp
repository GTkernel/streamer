//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "processor/processor.h"

#include <sstream>
#include <stdexcept>

#include "common/types.h"
#include "utils/utils.h"

static const size_t SLIDING_WINDOW_SIZE = 25;

Processor::Processor(ProcessorType type,
                     const std::vector<std::string>& source_names,
                     const std::vector<std::string>& sink_names)
    : num_frames_processed_(0),
      avg_processing_latency_ms_(0),
      processing_latencies_sum_ms_(0),
      trailing_avg_processing_latency_ms_(0),
      queue_latency_sum_ms_(0),
      type_(type),
      block_on_push_(false) {
  found_last_frame_ = false;
  stopped_ = true;

  for (const auto& source_name : source_names) {
    sources_.insert({source_name, nullptr});
    source_frame_cache_[source_name] = nullptr;
  }

  for (const auto& sink_name : sink_names) {
    sinks_.insert({sink_name, StreamPtr(new Stream)});
  }

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

void Processor::SetSink(const std::string& name, StreamPtr stream) {
  sinks_[name] = stream;
}

StreamPtr Processor::GetSink(const std::string& name) {
  if (sinks_.find(name) == sinks_.end()) {
    std::ostringstream msg;
    msg << "\"" << name << "\" is not a valid sink for processor \""
        << GetStringForProcessorType(GetType()) << "\".";
    throw std::runtime_error(msg.str());
  }
  return sinks_.at(name);
}

void Processor::SetSource(const std::string& name, StreamPtr stream) {
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

bool Processor::Start(size_t buf_size) {
  LOG(INFO) << "Start called";
  CHECK(stopped_) << "Processor has already started";

  processor_timer_.Start();

  // Check sources are filled
  for (const auto& source : sources_) {
    CHECK(source.second != nullptr)
        << "Source \"" << source.first << "\" is not set.";
  }

  // Subscribe sources
  for (auto& source : sources_) {
    readers_.emplace(source.first, source.second->Subscribe(buf_size));
  }

  stopped_ = false;
  process_thread_ = std::thread(&Processor::ProcessorLoop, this);
  return true;
}

bool Processor::Stop() {
  LOG(INFO) << "Stopping " << GetName() << "...";
  if (stopped_) {
    LOG(WARNING) << "Stop() called on a Processor that was already stopped!";
    return true;
  }

  // This signals that the process thread should stop processing new frames.
  stopped_ = true;

  // Stop all sink streams, which wakes up any blocking calls to
  // Stream::PushFrame() in the process thread.
  for (const auto& p : sinks_) {
    p.second->Stop();
  }

  // Unsubscribe from the source streams, which wakes up any blocking calls to
  // StreamReader::PopFrame() in the process thread.
  for (const auto& reader : readers_) {
    reader.second->UnSubscribe();
  }

  // Join the process thread, completing the main processing loop.
  process_thread_.join();

  // Do any processor-specific cleanup.
  bool result = OnStop();

  // Deallocate the source StreamReaders.
  readers_.clear();

  LOG(INFO) << "Stopped " << GetName();
  return result;
}

void Processor::ProcessorLoopDirect() {
  CHECK(Init()) << "Processor is not able to be initialized";
  while (!stopped_ && !found_last_frame_) {
    Process();
    ++num_frames_processed_;
  }
}

void Processor::ProcessorLoop() {
  CHECK(Init()) << "Processor is not able to be initialized";
  Timer local_timer;
  while (!stopped_ && !found_last_frame_) {
    // Cache source frames
    source_frame_cache_.clear();
    for (auto& reader : readers_) {
      auto source_name = reader.first;
      auto source_stream = reader.second;

      auto frame = source_stream->PopFrame();
      if (frame == nullptr) {
        // The only way for PopFrame() to return a nullptr when called without a
        // a timeout is if Stop() was called on the StreamReader. That should
        // only happen if this processor is being stopped. Therefore, we should
        // just return.
        return;
      } else if (frame->IsStopFrame()) {
        // This frame is signaling the pipeline to stop. We need to forward
        // it to our sinks, then not process it or any future frames.
        for (const auto& p : sinks_) {
          PushFrame(p.first, std::make_unique<Frame>(frame));
        }
        return;
      } else {
        // Calculate queue latency
        double start_ms = frame->GetValue<double>("start_time_ms");
        double end_ms = Context::GetContext().GetTimer().ElapsedMSec();
        queue_latency_sum_ms_ += end_ms - start_ms;
        source_frame_cache_[source_name] = std::move(frame);
      }
    }

    processing_start_micros_ = boost::posix_time::microsec_clock::local_time();
    Process();
    double processing_latency_ms =
        (double)(boost::posix_time::microsec_clock::local_time() -
                 processing_start_micros_)
            .total_microseconds();
    processing_start_micros_ = boost::posix_time::not_a_date_time;

    ++num_frames_processed_;

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

double Processor::GetHistoricalProcessFps() {
  return num_frames_processed_ / (processor_timer_.ElapsedMSec() / 1000);
}

double Processor::GetAvgProcessingLatencyMs() const {
  return avg_processing_latency_ms_;
}

double Processor::GetAvgQueueLatencyMs() const {
  return queue_latency_sum_ms_ / num_frames_processed_;
}

ProcessorType Processor::GetType() const { return type_; }

std::string Processor::GetName() const {
  return GetStringForProcessorType(GetType());
}

zmq::socket_t* Processor::GetControlSocket() { return control_socket_; };

void Processor::SetBlockOnPush(bool block) { block_on_push_ = block; }

void Processor::PushFrame(const std::string& sink_name,
                          std::unique_ptr<Frame> frame) {
  CHECK(sinks_.count(sink_name) != 0)
      << GetStringForProcessorType(GetType())
      << " does not have a sink named \"" << sink_name << "\"!";
  if (!processing_start_micros_.is_not_a_date_time()) {
    frame->SetValue(GetName() + ".total_micros",
                    boost::posix_time::microsec_clock::local_time() -
                        processing_start_micros_);
  }
  if (frame->IsStopFrame()) {
    found_last_frame_ = true;
  }
  sinks_[sink_name]->PushFrame(std::move(frame), block_on_push_);
}

std::unique_ptr<Frame> Processor::GetFrame(const std::string& source_name) {
  if (source_frame_cache_.find(source_name) == source_frame_cache_.end()) {
    std::ostringstream msg;
    msg << "\""
        << "\" is not a valid source for processor \""
        << GetStringForProcessorType(GetType()) << "\".";
    throw std::out_of_range(msg.str());
  }
  return std::move(source_frame_cache_[source_name]);
}

std::unique_ptr<Frame> Processor::GetFrameDirect(
    const std::string& source_name) {
  if (readers_.find(source_name) == readers_.end()) {
    std::ostringstream msg;
    msg << "\""
        << "\" is not a valid source for processor \""
        << GetStringForProcessorType(GetType()) << "\".";
    throw std::out_of_range(msg.str());
  }
  return readers_.at(source_name)->PopFrame();
}
