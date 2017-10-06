
#ifndef STREAMER_PROCESSOR_THROTTLER_H_
#define STREAMER_PROCESSOR_THROTTLER_H_

#include <memory>

#include "common/types.h"
#include "processor/processor.h"
#include "utils/time_utils.h"

// A processor that a stream's framerate to a specified maximum.
class Throttler : public Processor {
 public:
  Throttler(double fps);
  static std::shared_ptr<Throttler> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

  void SetFps(double fps);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  double delay_ms_;
  Timer timer_;
};

#endif  // STREAMER_PROCESSOR_THROTTLER_H_
