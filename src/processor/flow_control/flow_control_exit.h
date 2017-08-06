
#ifndef STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_
#define STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_

#include <memory>

#include "common/types.h"
#include "processor/processor.h"

// FlowControlExit is used to update the token count maintained by
// FlowControlEntrance when frames leave the pipeline. Both classes together are
// used to limit the number of outstanding frames in the pipeline.
class FlowControlExit : public Processor {
 public:
  FlowControlExit();
  static std::shared_ptr<FlowControlExit> Create(
      const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;
};

#endif  // STREAMER_PROCESSOR_FLOW_CONTROL_FLOW_CONTROL_EXIT_H_
