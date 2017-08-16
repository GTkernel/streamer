
#ifndef STREAMER_PROCESSOR_STRIDER_H_
#define STREAMER_PROCESSOR_STRIDER_H_

#include <memory>
#include <mutex>

#include "common/types.h"
#include "processor/processor.h"

// Strider passes through frames at a user-specific stride
class Strider : public Processor {
 public:
  Strider(unsigned int stride);
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
  unsigned int stride_;
  unsigned int num_frames_processed_ = 0;
};

#endif  // STREAMER_PROCESSOR_STRIDER_H_
