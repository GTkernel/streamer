
#ifndef STREAMER_PROCESSOR_STRIDER_H_
#define STREAMER_PROCESSOR_STRIDER_H_

#include <memory>
#include <mutex>

#include "common/types.h"
#include "processor/processor.h"

// Strider performs admission control of frames to limit the number
// of outstanding frames in the pipeline. It should be used together with
// FlowControlExit.
class Strider : public Processor {
 public:
  // "max_tokens" should not be larger than the capacity of the shortest stream
  // queue in the flow control domain. This is to ensure that no frames are
  // dropped due to queue overflow.
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
  // Used to verify that num_tokens_available_ never exceeds the original number
  // of tokens.
  unsigned int stride_;
  unsigned int num_frames_processed_ = 0;
};

#endif  // STREAMER_PROCESSOR_STRIDER_H_
