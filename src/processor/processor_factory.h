
#ifndef STREAMER_PROCESSOR_PROCESSOR_FACTORY_H_
#define STREAMER_PROCESSOR_PROCESSOR_FACTORY_H_

#include "common/types.h"
#include "processor/processor.h"

class ProcessorFactory {
 public:
  static std::shared_ptr<Processor> Create(ProcessorType type,
                                           FactoryParamsType params);
};

#endif  // STREAMER_PROCESSOR_PROCESSOR_FACTORY_H_
