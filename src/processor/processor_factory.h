//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "camera/camera.h"
#include "processor.h"

#ifndef STREAMER_PROCESSOR_FACTORY_H
#define STREAMER_PROCESSOR_FACTORY_H

class ProcessorFactory {
 public:
  static std::shared_ptr<Processor> CreateInstance(ProcessorType processor_type,
                                                   FactoryParamsType params);

 private:
  // Note: It is fine to use raw pointer here, it will be wrapped in a
  // shared_ptr by CreateInstance()
  // TODO: maintain a unified pattern
  static std::shared_ptr<Camera> CreateCamera(const FactoryParamsType &params);
  static Processor *CreateCustomProcessor(const FactoryParamsType &params);
  static Processor *CreateDummyNNProcessor(const FactoryParamsType &params);
  static Processor *CreateEncoder(const FactoryParamsType &params);
  static Processor *CreateFileWriter(const FactoryParamsType &params);
#ifdef USE_RPC
  static Processor *CreateFrameReceiver(const FactoryParamsType &params);
  static Processor *CreateFrameSender(const FactoryParamsType &params);
#endif
  static Processor *CreateImageClassifier(const FactoryParamsType &params);
  static Processor *CreateImageSegmenter(const FactoryParamsType &params);
  static Processor *CreateImageTransformer(const FactoryParamsType &params);
  static Processor *CreateOpenCVFaceDetector(const FactoryParamsType &params);
#ifdef USE_ZMQ
  static Processor *CreateStreamPublisher(const FactoryParamsType &params);
#endif
};

#endif  // STREAMER_PROCESSOR_FACTORY_H
