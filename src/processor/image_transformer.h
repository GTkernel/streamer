//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_IMAGETRANSFORMPROCESSOR_H
#define STREAMER_IMAGETRANSFORMPROCESSOR_H

#include "common/types.h"
#include "processor.h"
#include "stream/stream.h"

enum CropType { CROP_TYPE_INVALID = 0, CROP_TYPE_CENTER = 1 };

class ImageTransformer : public Processor {
 public:
  ImageTransformer(std::shared_ptr<Stream> input_stream,
                   const Shape &target_shape, CropType crop_type,
                   bool subtract_mean = true);
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  Shape target_shape_;
  cv::Mat mean_image_;
  CropType crop_type_;
  bool subtract_mean_;

  // Temporary mat for image processing, reduce memory (de)allocation
  cv::Mat sample_image_;
  cv::Mat sample_resized_;
  cv::Mat sample_cropped_;
  cv::Mat sample_float_;
  cv::Mat sample_normalized_;
};

#endif  // STREAMER_IMAGETRANSFORMPROCESSOR_H
