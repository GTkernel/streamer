
#ifndef STREAMER_PROCESSOR_STRIDER_H_
#define STREAMER_PROCESSOR_STRIDER_H_

#include <memory>

#include "common/types.h"
#include "processor/processor.h"

// The Strider processor passes one out of every "stride" frames and drops all
// other frames.
class Strider : public Processor {
 public:
  Strider(unsigned long stride);
  static std::shared_ptr<Strider> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  unsigned long stride_;
  unsigned long num_frames_processed_;
};

#endif  // STREAMER_PROCESSOR_STRIDER_H_
