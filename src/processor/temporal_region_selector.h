
#ifndef STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_
#define STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_

#include <string>

#include "common/types.h"
#include "processor/processor.h"

// A processor that selects a region of frames based on their frame id.
class TemporalRegionSelector : public Processor {
 public:
  TemporalRegionSelector(unsigned long start_id, unsigned long end_id);
  static std::shared_ptr<TemporalRegionSelector> Create(
      const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  unsigned long start_id_;
  unsigned long end_id_;
};

#endif  // STREAMER_PROCESSOR_TEMPORAL_REGION_SELECTOR_H_
