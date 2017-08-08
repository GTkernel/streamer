//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
#define STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_

#include "common/types.h"
#include "processor/processor.h"
#include "stream/stream.h"

class ImageTransformer : public Processor {
 public:
  ImageTransformer(const Shape& target_shape);

  static std::shared_ptr<ImageTransformer> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  Shape target_shape_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
