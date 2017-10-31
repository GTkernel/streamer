#ifndef STREAMER_PROCESSOR_DISPLAY_H_
#define STREAMER_PROCESSOR_DISPLAY_H_

#include <memory>

#include "common/types.h"
#include "processor/processor.h"

// The Display processor displays frames at a specified zoom and angle
class Display : public Processor {
 public:
  Display(std::string key, unsigned int angle, float zoom,
          std::string window_name_);
  static std::shared_ptr<Display> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string key_;
  unsigned int angle_;
  float zoom_;
  std::string window_name_;
};

#endif  // STREAMER_PROCESSOR_DISPLAY_H_
