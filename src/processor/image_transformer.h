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
  ImageTransformer(const Shape& target_shape, bool crop,
                   unsigned int angle = 0);

  static std::shared_ptr<ImageTransformer> Create(
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
  Shape target_shape_;
  bool crop_;
  bool convert_;
  unsigned int angle_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
