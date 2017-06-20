
#ifndef STREAMER_PROCESSOR_THROTTLER_H_
#define STREAMER_PROCESSOR_THROTTLER_H_

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"

/**
 * @brief A processor that limits the rate of frames being passed through.
 */
class Throttler : public Processor {
 public:
  Throttler(int fps);
  void SetFps(int fps);
  static std::shared_ptr<Throttler> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  int fps_;
  unsigned long long delay_ms_;
  Timer timer_;
};

#endif  // STREAMER_PROCESSOR_THROTTLER_H_
