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
  ImageTransformer(const Shape& target_shape, bool crop, bool convert,
                   bool subtract_mean = true);

  static std::shared_ptr<ImageTransformer> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  Shape target_shape_;
  cv::Mat mean_image_;
  bool crop_;
  bool convert_;
  bool subtract_mean_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
