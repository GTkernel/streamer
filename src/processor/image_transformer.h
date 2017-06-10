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
  ImageTransformer(const Shape& target_shape, bool subtract_mean = true);

  static std::shared_ptr<ImageTransformer> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  Shape target_shape_;
  cv::Mat mean_image_;
  bool subtract_mean_;

  // Temporary mat for image processing, reduce memory (de)allocation
  cv::Mat sample_image_;
  cv::Mat sample_resized_;
  cv::Mat sample_cropped_;
  cv::Mat sample_float_;
  cv::Mat sample_normalized_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
