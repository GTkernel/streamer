//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "dummy_nn_processor.h"
#include "image_classifier.h"
#include "image_segmenter.h"
#include "image_transformer.h"
#include "opencv_face_detector.h"

#ifndef STREAMER_PROCESSOR_FACTORY_H
#define STREAMER_PROCESSOR_FACTORY_H

class ProcessorFactory {
 public:
  static std::shared_ptr<Processor> CreateInstance(ProcessorType processor_type,
                                                   FactoryParamsType params);

 private:
  // Note: It is fine to use raw pointer here, it will be wrapped in a
  // shared_ptr by CreateInstance()
  static Processor *CreateCustomProcessor(const FactoryParamsType &params);
  static Processor *CreateCamera(const FactoryParamsType &params);
  static Processor *CreateEncoder(const FactoryParamsType &params);
  static Processor *CreateImageClassifier(const FactoryParamsType &params);
  static Processor *CreateImageSegmenter(const FactoryParamsType &params);
  static Processor *CreateImageTransformer(const FactoryParamsType &params);
  static Processor *CreateOpenCVFaceDetector(const FactoryParamsType &params);
  static Processor *CreateDummyNNProcessor(const FactoryParamsType &params);
};

#endif  // STREAMER_PROCESSOR_FACTORY_H
