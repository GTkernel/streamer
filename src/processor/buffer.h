
#ifndef STREAMER_PROCESSOR_BUFFER_H_
#define STREAMER_PROCESSOR_BUFFER_H_

#include <memory>

#include <boost/circular_buffer.hpp>

#include "common/types.h"
#include "processor/processor.h"

// This processor stores a configurable number of recent frames, effectively
// introducing a delay in the pipeline.
class Buffer : public Processor {
 public:
  Buffer(unsigned long num_frames);
  static std::shared_ptr<Buffer> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  boost::circular_buffer<std::unique_ptr<Frame>> buffer_;
};

#endif  // STREAMER_PROCESSOR_BUFFER_H_
